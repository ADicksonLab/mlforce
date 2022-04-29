#ifndef REFERENCE_PY_TORCH_KERNELS_H_
#define REFERENCE_PY_TORCH_KERNELS_H_

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

#include "PyTorchKernels.h"
#include "Hungarian.h"
//#include "Distances.h"
#include "openmm/Platform.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstring>

using namespace std;
using namespace torch::indexing;
static const std::vector<string> PARAMETERNAMES={"charge_g", "sigma_g", "epsilon_g", "lambda_g"};



namespace PyTorchPlugin {

/**
 * This kernel is invoked by PyTorchForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcPyTorchForceKernel : public CalcPyTorchForceKernel {
public:
	/**
	 * Construct a new ReferenceCalcPyTorchForceKernel object
	 *
	 * @param name
	 * @param platform
	 */

	ReferenceCalcPyTorchForceKernel(std::string name, const OpenMM::Platform& platform) : CalcPyTorchForceKernel(name, platform) {
	}
	/**
	 *  Destroy the ReferenceCalcPyTorchForceKernel object
	 *
	 */
	~ReferenceCalcPyTorchForceKernel();
	/**
	 * Initialize the kernel.
	 * Gets the loaded Pytorch ScriptModule and moves on CPU.
	 * Copy the Ghost particle indices, target features, and dynamical variables vectors to tensors.
	 *
	 * @param system the System this kernel will be applied to
	 * @param force  the PyTorchForce this kernel will be used for
	 * @param nnModule the PyTorch module to use for computing forces and energy
	 */

	void initialize(const OpenMM::System& system, const PyTorchForce& force,
			torch::jit::script::Module nnModule);
	/**
	* Execute the kernel to calculate the forces and/or energy.
	*
	* Extract the system's positions and forces and move the ghost atoms' positions to a tensor.
	* Read the dynamical variable values (signals) from the Context and moves them to a tensor.
	* Extract the charges of the ghost atoms from the dynamical variables tensor (first column of the signals).
	* Create the input vector of ghost atoms' positions snd charges.
	* Run the Pytorch model (TorchANI) to get the AEVs.
	* Combine the AEVs and signals to get features.
	* Calculates the all to all distances between the ghost atom features and the target ligand features. The
	* 'ghFeaturesTensor.index({Slice(), None})' repeats each ghost atom features n times where n is the number
	* of ghost particles. The assumption is that we have the same number of atoms for ghost and lignad.
	* Then Euclidian distances (vector norm 2) are calculated.
	* The distance matrix is passed to the Hungarian algorithm to get the optimal mapping between the ghost
	* atoms and the target ligand.
	* The energy is calculated by determining the Mean Squared Erro (mse) between the AVEs of ghost atoms and
	* target are determined and multiplied by the scale.
	* The forces on dynamical variables are simply the weighted difference between the ghost atoms and
	* the target ligand signals
	* The loss value (energy) back-propagated through model.
	* The forces on ghost atoms are determined as the negative gradient of the energy value with respect to the positions
	* The system forces get updated by adding the ghost atoms forces
	*
	* @param context the context in which to execute this kernel
	* @param includeForces true if forces should be calculated
	* @param includeEnergy true if the energy should be calculated
	* @return the potential energy due to the force
	*/
	double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

private:
	torch::jit::script::Module nnModule;
	torch::Tensor boxVectorsTensor;
	torch::Tensor targetFeaturesTensor;
	std::vector<int> particleIndices;
	std::vector<double> signalForceWeights;
	std::vector<std::vector<double>> targetFeatures;
	double scale;
	bool usePeriodic;
	HungarianAlgorithm hungAlg;
};

} // namespace PyTorchPlugin

#endif /*REFERENCE_NEURAL_NETWORK_KERNELS_H_*/
