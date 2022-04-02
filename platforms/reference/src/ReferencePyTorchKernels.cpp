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

#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "Hungarian.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

/**
 * @brief
 *
 * @param context
 * @return vector<Vec3>&
 */
static vector<Vec3>& extractPositions(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((vector<Vec3>*) data->positions);
}

/**
 * @brief
 *
 * @param context
 * @return vector<Vec3>&
 */
static vector<Vec3>& extractForces(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((vector<Vec3>*) data->forces);
}
/**
 * @brief
 *
 * @param context
 * @return Vec3*
 */
static Vec3* extractBoxVectors(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return (Vec3*) data->periodicBoxVectors;
}

/**
 * @brief
 *
 * @param context
 * @return map<string, double>&
 */
static map<string, double>& extractEnergyParameterDerivatives(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((map<string, double>*) data->energyParameterDerivatives);
}


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

ReferenceCalcPyTorchForceKernel::~ReferenceCalcPyTorchForceKernel() {
}


/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */
void ReferenceCalcPyTorchForceKernel::initialize(const System& system, const PyTorchForce& force, torch::jit::script::Module nnModule) {
	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();

	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

	//get target features
	targetFeatures = force.getTargetFeatures();
	targetFeaturesTensor = torch::zeros({static_cast<int64_t>(targetFeatures.size()),
		static_cast<int64_t>(targetFeatures[0].size())},
		torch::TensorOptions().dtype(torch::kFloat64));

	for (std::size_t i = 0; i < targetFeatures.size(); i++)
		targetFeaturesTensor.slice(0, i, i+1) = torch::from_blob(targetFeatures[i].data(),
			{(long long)targetFeatures[0].size()},
			torch::TensorOptions().dtype(torch::kFloat64));

	if (usePeriodic) {
	  int64_t boxVectorsDims[] = {3, 3};
	  boxVectorsTensor = torch::zeros(boxVectorsDims);
	  boxVectorsTensor = boxVectorsTensor.to(torch::kFloat64);
	}

}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double ReferenceCalcPyTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

	// Get the  positions from the context (previous step)
	vector<Vec3>& MDPositions = extractPositions(context);
	vector<Vec3>& MDForce = extractForces(context);

	int numGhostParticles = particleIndices.size();

	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3},
		torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64));


	auto positions = positionsTensor.accessor<double, 2>();

	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0] = MDPositions[particleIndices[i]][0];
		positions[i][1] = MDPositions[particleIndices[i]][1];
		positions[i][2] = MDPositions[particleIndices[i]][2];
	}

	torch::Tensor signalsTensor = torch::zeros({numGhostParticles, 4}, torch::kFloat64);
	std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);
	signalsTensor = torch::from_blob(globalVariables.data(),
		{static_cast<int64_t>(numGhostParticles), 4}, torch::kFloat64);

	// Run the pytorch model and get the energy
	auto charges = signalsTensor.index({Slice(), 0});
	vector<torch::jit::IValue> nnInputs = {positionsTensor, charges};

	// Copy the box vector
	if (usePeriodic) {
		Vec3* box = extractBoxVectors(context);
		torch::Tensor boxVectorsTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
		nnInputs.push_back(boxVectorsTensor);
	}

	// outputTensor : attributes (ANI AEVs)
	torch::Tensor outputTensor = nnModule.forward(nnInputs).toTensor();

	// concat ANI AEVS with atomic attributes [charge, sigma, epsiolon, lambda]
	torch::Tensor ghFeaturesTensor = torch::cat({outputTensor, signalsTensor}, 1);
	//std::cout<<ghFeaturesTensor <<"\n";

	torch::Tensor distMatTensor = at::norm(ghFeaturesTensor.index({Slice(), None})
		- targetFeaturesTensor, 2, 2);


	//convert it to a 2d vector
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
	torch::Tensor energyTensor = scale * torch::mse_loss(outputTensor,
		reFeaturesTensor.narrow(1, 0, outputTensor.size(1))).clone();

	// calculate force on the signals clips out singals from the end of features
	torch::Tensor targtSignalsTensor = reFeaturesTensor.narrow(1, -4, 4);

	// update the global variables derivatives
	map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);
	auto targetSignalsData = targtSignalsTensor.accessor<double, 2>();
	double parameter_deriv;
	// double target_sig;
	for (int i = 0; i < numGhostParticles; i++) {
		for (int j=0; j<4; j++)
		{
			//target_sig = targtSignalsTensor.data_ptr<double>()[i,j];
			parameter_deriv = signalForceWeights[j] * (globalVariables[i*4+j] - targetSignalsData[i][j]);
			energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += parameter_deriv;
		}
	}
	// get forces on positions as before

	if (includeForces) {
		energyTensor.backward();

		// check if positions have gradients
		auto forceTensor = torch::zeros_like(positionsTensor);

		forceTensor = - positionsTensor.grad();
		positionsTensor.grad().zero_();
		if (!(forceTensor.dtype() == torch::kFloat64))
			forceTensor = forceTensor.to(torch::kFloat64);

		auto NNForce = forceTensor.accessor<double, 2>();
		for (int i = 0; i < numGhostParticles; i++) {
			MDForce[particleIndices[i]][0] += NNForce[i][0];
			MDForce[particleIndices[i]][1] += NNForce[i][1];
			MDForce[particleIndices[i]][2] += NNForce[i][2];
		}
	}
	return energyTensor.item<double>();
  }
