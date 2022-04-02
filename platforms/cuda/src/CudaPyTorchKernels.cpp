/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaPyTorchKernels.h"
#include "CudaPyTorchKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <cuda_runtime_api.h>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;


/**
 * @brief
 *
 * @param context
 * @param numParticles
 * @return std::vector<double>
 */

std::vector<double> extractContextVariables(ContextImpl& context, int numParticles) {
	std::vector<double> signals;
	string name;
	for (int i=0; i < numParticles; i++) {
		for (std::size_t j=0; j < PARAMETERNAMES.size(); j++) {
			signals.push_back(context.getParameter(PARAMETERNAMES[j]+std::to_string(i)));
		}
	}
	return signals;
}

/**
 * @brief
 *
 * @param ptr
 * @param nRows
 * @param nCols
 * @return std::vector<std::vector<double>>
 */
std::vector<std::vector<double>> tensorTo2DVec(double* ptr, int nRows, int nCols) {
	std::vector<std::vector<double>> distMat(nRows, std::vector<double>(nCols));
	for (int i=0; i<nRows; i++) {
		std::vector<double> vec(ptr+nCols*i, ptr+nRows*(i+1));
		distMat[i] = vec;
	}
	return distMat;
}

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix) \
if (result != CUDA_SUCCESS) { \
	std::stringstream m; \
	m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
	throw OpenMMException(m.str());\
}

/**
 * @brief Destroy the Cuda CalcPy Torch Force Kernel:: Cuda CalcPy Torch Force Kernel object
 *
 */
CudaCalcPyTorchForceKernel::~CudaCalcPyTorchForceKernel() {
}

/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */
void CudaCalcPyTorchForceKernel::initialize(const System& system, const PyTorchForce& force, torch::jit::script::Module nnModule) {
	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	usePeriodic = force.usesPeriodicBoundaryConditions();
	scale = force.getScale();
	particleIndices = force.getParticleIndices();
	usePeriodic = force.usesPeriodicBoundaryConditions();
	signalForceWeights = force.getSignalForceWeights();
	int numGhostParticles = particleIndices.size();

	std::vector<std::vector<double>> targetFeatures = force.getTargetFeatures();
	targetFeaturesTensor = torch::zeros({static_cast<int64_t>(targetFeatures.size()),
		static_cast<int64_t>(targetFeatures[0].size())},
		torch::kFloat64);

	for (std::size_t i = 0; i < targetFeatures.size(); i++)
		targetFeaturesTensor.slice(0, i, i+1) = torch::from_blob(targetFeatures[i].data(),
			{static_cast<int64_t>(targetFeatures[0].size())},
			torch::TensorOptions().dtype(torch::kFloat64));


	torch::TensorOptions options = torch::TensorOptions().
		device(torch::kCPU).
		dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	targetFeaturesTensor = targetFeaturesTensor.to(options);

	if (usePeriodic) {
		boxVectorsTensor = torch::empty({3, 3}, options);
	}

	// Inititalize CUDA objects.

	cu.setAsCurrent();
	map<string, string> defines;
	CUmodule program = cu.createModule(CudaPyTorchKernelSources::PyTorchForce, defines);
	copyInputsKernel = cu.getKernel(program, "copyInputs");
	addForcesKernel = cu.getKernel(program, "addForces");

}

double CudaCalcPyTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

	int numParticles = cu.getNumAtoms();
	int numGhostParticles = particleIndices.size();
	vector<Vec3> MDPositions;
	context.getPositions(MDPositions);
	torch::Tensor positionsTensor = torch::empty({static_cast<int64_t>(numGhostParticles), 3},
		cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	if (cu.getUseDoublePrecision()) {
		auto positions = positionsTensor.accessor<double, 2>();
		for (int i = 0; i < numGhostParticles; i++) {
			positions[i][0] = MDPositions[particleIndices[i]][0];
			positions[i][1] = MDPositions[particleIndices[i]][1];
			positions[i][2] = MDPositions[particleIndices[i]][2];
		}
	}
	else {
		auto positions = positionsTensor.accessor<float, 2>();
		for (int i = 0; i < numGhostParticles; i++) {
			positions[i][0] = MDPositions[particleIndices[i]][0];
			positions[i][1] = MDPositions[particleIndices[i]][1];
			positions[i][2] = MDPositions[particleIndices[i]][2];
		}
	}

	torch::Tensor signalsTensor = torch::empty({numGhostParticles, 4}, torch::kFloat64);

	std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);
	signalsTensor = torch::from_blob(globalVariables.data(),
		{static_cast<int64_t>(numGhostParticles), 4}, torch::kFloat64);

	torch::TensorOptions options = torch::TensorOptions().device(torch::kCPU)
		.dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	signalsTensor = signalsTensor.to(options);
	positionsTensor = positionsTensor.to(options);
	positionsTensor.requires_grad_(true);
	auto charges = signalsTensor.index({Slice(), 0});
	vector<torch::jit::IValue> nnInputs = {positionsTensor, charges};
	if (usePeriodic) {
	  Vec3 box[3];
	  cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
	  boxVectorsTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);

	  boxVectorsTensor = boxVectorsTensor.to(options);
	  nnInputs.push_back(boxVectorsTensor);
	}

	// synchronizing the current context before switching to PyTorch
	CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");

	torch::Tensor outputTensor = nnModule.forward(nnInputs).toTensor();

	// concat ANI AEVS with atomic attributes [charge, sigma, epsiolon, lambda]
	torch::Tensor ghFeaturesTensor = torch::cat({outputTensor, signalsTensor}, 1);

	torch::Tensor distMatTensor = at::norm(ghFeaturesTensor.index({Slice(), None})
		- targetFeaturesTensor, 2, 2);


	//convert it to a 2d vector
	if (!cu.getUseDoublePrecision())
		distMatTensor=distMatTensor.to(torch::kFloat64);

	std::vector<std::vector<double>> distMatrix = tensorTo2DVec(distMatTensor.data_ptr<double>(),
		numGhostParticles,
		static_cast<int>(targetFeaturesTensor.size(0)));

	// call Hungarian algorithm to determine mapping (and loss)
	vector<int> assignment;
	assignment = hungAlg.Solve(distMatrix);

	// Save the assignments in the context variables
	for (std::size_t i=0; i<assignment.size(); i++) {
		context.setParameter("assignment_g"+std::to_string(i), assignment[i]);
	}

	// reorder the targetFeaturesTensor using the mapping
	torch::Tensor reFeaturesTensor = targetFeaturesTensor.index({{torch::tensor(assignment)}}).clone();
	//select ANI faetures


	// get forces on positions as before
	torch::Tensor energyTensor = scale * torch::mse_loss(outputTensor,
		reFeaturesTensor.narrow(1, 0, outputTensor.size(1))).clone();


	// calculate force on the signals clips out singals from the end of features
	torch::Tensor targtSignalsTensor = reFeaturesTensor.narrow(1, -4, 4);
	// update the global variables derivatives

	map<string, double> &energyParamDerivs = cu.getEnergyParamDerivWorkspace();

	if (cu.getUseDoublePrecision()) {
		double parameter_deriv;
		auto targetSignalsData = targtSignalsTensor.accessor<double, 2>();
		for (int i = 0; i < numGhostParticles; i++) {
			for (int j=0; j<4; j++){
				parameter_deriv = signalForceWeights[j] * (globalVariables[i*4+j] - targetSignalsData[i][j]);
				energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += parameter_deriv;
			}
		}
	}
	else {
		float parameter_deriv;
		auto targetSignalsData = targtSignalsTensor.accessor<float, 2>();
		for (int i = 0; i < numGhostParticles; i++) {
			for (int j=0; j<4; j++) {
				parameter_deriv = signalForceWeights[j] * (globalVariables[i*4+j] - targetSignalsData[i][j]);
				energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += parameter_deriv;
			}
		}
	}

	if (includeForces) {
		energyTensor.backward();
		auto forceTensor = torch::zeros_like(positionsTensor);
		forceTensor = - positionsTensor.grad().clone();
		positionsTensor.grad().zero_();
		torch::Tensor paddedForceTensor = torch::zeros({numParticles, 3}, options);
		paddedForceTensor.narrow(0,
			static_cast<int64_t>(particleIndices[0]),
			static_cast<int64_t>(particleIndices.size())).copy_(forceTensor);

		paddedForceTensor = paddedForceTensor.to(torch::kCUDA);
		CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
		cu.setAsCurrent();
		void* fdata;
		if (cu.getUseDoublePrecision()) {
			if (!(paddedForceTensor.dtype() == torch::kFloat64))
				paddedForceTensor = paddedForceTensor.to(torch::kFloat64);
			fdata = paddedForceTensor.data_ptr<double>();
		}
		else {
			if (!(paddedForceTensor.dtype() == torch::kFloat32))
				paddedForceTensor= paddedForceTensor.to(torch::kFloat32);
			fdata = paddedForceTensor.data_ptr<float>();
		}
		int paddedNumAtoms = cu.getPaddedNumAtoms();
		void* forceArgs[] = {&fdata, &cu.getForce().getDevicePointer(),
			&cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
			cu.executeKernel(addForcesKernel, forceArgs, numParticles);

	}
	return energyTensor.item<double>();
}
