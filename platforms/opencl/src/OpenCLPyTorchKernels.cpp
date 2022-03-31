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

#include "OpenCLPyTorchKernels.h"
#include "OpenCLPyTorchKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

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

/**
 * @brief Destroy the OpenCLCalcPyTorch Force Kernel:: OpenCLCalcPyTorch Force Kernel object
 *
 */

OpenCLCalcPyTorchForceKernel::~OpenCLCalcPyTorchForceKernel() {}

/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */

void OpenCLCalcPyTorchForceKernel::initialize(const System& system, const PyTorchForce& force,
											  torch::jit::script::Module nnModule){

	this->nnModule = nnModule;
	nnModule.eval();

	usePeriodic = force.usesPeriodicBoundaryConditions();
	scale = force.getScale();
	particleIndices = force.getParticleIndices();
	usePeriodic = force.usesPeriodicBoundaryConditions();
	signalForceWeights = force.getSignalForceWeights();
	int numGhostParticles = particleIndices.size();

	//get target features
	std::vector<std::vector<double>> targetFeatures = force.getTargetFeatures();
	targetFeaturesTensor = torch::zeros({static_cast<int64_t>(targetFeatures.size()),
		static_cast<int64_t>(targetFeatures[0].size())},
		torch::kFloat64);

	for (std::size_t i = 0; i < targetFeatures.size(); i++)
		targetFeaturesTensor.slice(0, i, i+1) = torch::from_blob(targetFeatures[i].data(),
			{static_cast<int64_t>(targetFeatures[0].size())},
			torch::TensorOptions().dtype(torch::kFloat64));


	if (!cl.getUseDoublePrecision())
		targetFeaturesTensor = targetFeaturesTensor.to(torch::kFloat32);

	if (usePeriodic)
		boxVectorsTensor = torch::empty({3, 3}, torch::kFloat64);

	// Inititalize OpenCL objects.
	int numParticles = system.getNumParticles();
	map<string, string> defines;
	if (cl.getUseDoublePrecision()) {
	  networkForces.initialize<double>(cl, 3*numParticles, "networkForces");
	  defines["FORCES_TYPE"] = "double";
	}
	else {
	  networkForces.initialize<float>(cl, 3*numParticles, "networkForces");
	  defines["FORCES_TYPE"] = "float";
	}
	cl::Program program = cl.createProgram(OpenCLPyTorchKernelSources::PyTorchForce, defines);
	addForcesKernel = cl::Kernel(program, "addForces");
	}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */

double OpenCLCalcPyTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
	// Get the  positions from the context (previous step)
	int numParticles = cl.getNumAtoms();
	int numGhostParticles = particleIndices.size();
	vector<Vec3> MDPositions;
	context.getPositions(MDPositions);
	torch::Tensor positionsTensor = torch::empty({static_cast<int64_t>(numGhostParticles), 3},
		cl.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	if (cl.getUseDoublePrecision()) {
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


	if (!cl.getUseDoublePrecision()) {

	  signalsTensor = signalsTensor.to(torch::kFloat32);
	  positionsTensor = positionsTensor.to(torch::kFloat32);
	}

	positionsTensor.requires_grad_(true);

	// Run the pytorch model and get the energy
	vector<torch::jit::IValue> nnInputs = {positionsTensor};

	if (usePeriodic) {
	  Vec3 box[3];
	  cl.getPeriodicBoxVectors(box[0], box[1], box[2]);
	  boxVectorsTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
	  if (!cl.getUseDoublePrecision())
		boxVectorsTensor = boxVectorsTensor.to(torch::kFloat32);
	  nnInputs.push_back(boxVectorsTensor);
	}

	torch::Tensor outputTensor = nnModule.forward(nnInputs).toTensor();
	torch::Tensor ghFeaturesTensor = torch::cat({outputTensor, signalsTensor}, 1);


	torch::Tensor distMatTensor = at::norm(ghFeaturesTensor.index({Slice(), None})
		- targetFeaturesTensor, 2, 2);

	//convert it to a 2d vector
	if (!cl.getUseDoublePrecision())
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
	torch::Tensor targtSignalsTensor = reFeaturesTensor.narrow(1, -4, 4).clone();
	// update the global variables derivatives
	map<string, double> &energyParamDerivs = cl.getEnergyParamDerivWorkspace();
	if (cl.getUseDoublePrecision()) {
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

	if (includeForces){
		energyTensor.backward();
		auto forceTensor = torch::zeros_like(positionsTensor);
		forceTensor = - positionsTensor.grad();
		positionsTensor.grad().zero_();
		torch::Tensor paddedForceTensor = torch::zeros({numParticles, 3},
			cl.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

		paddedForceTensor.narrow(0,
			static_cast<int64_t>(particleIndices[0]),
	        static_cast<int64_t>(particleIndices.size())).copy_(forceTensor);


		if (cl.getUseDoublePrecision()) {
			if (!(paddedForceTensor.dtype() == torch::kFloat64))
				paddedForceTensor = paddedForceTensor.to(torch::kFloat64);
			double* data = paddedForceTensor.data_ptr<double>();
			networkForces.upload(data);
		}
		else {
			if (!(paddedForceTensor.dtype() == torch::kFloat32))
			  paddedForceTensor= paddedForceTensor.to(torch::kFloat32);

			float* data = paddedForceTensor.data_ptr<float>();
			networkForces.upload(data);
		}
		addForcesKernel.setArg<cl::Buffer>(0, networkForces.getDeviceBuffer());
		addForcesKernel.setArg<cl::Buffer>(1, cl.getForceBuffers().getDeviceBuffer());
		addForcesKernel.setArg<cl::Buffer>(2, cl.getAtomIndexArray().getDeviceBuffer());
		addForcesKernel.setArg<cl_int>(3, numParticles);
		cl.executeKernel(addForcesKernel, numParticles);
	}

	return energyTensor.item<double>();
}
