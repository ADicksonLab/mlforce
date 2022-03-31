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
	ReferenceCalcPyTorchForceKernel(std::string name, const OpenMM::Platform& platform) : CalcPyTorchForceKernel(name, platform) {
	}
	~ReferenceCalcPyTorchForceKernel();

	void initialize(const OpenMM::System& system, const PyTorchForce& force,
			torch::jit::script::Module nnModule);

	double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
  //std::vector<std::vector<double>>& tensorToVec(double* ptr, int nRows, int nCols);
private:
	torch::jit::script::Module nnModule;
	torch::Tensor boxVectorsTensor;
	torch::Tensor targetFeaturesTensor;
	std::vector<int> particleIndices;
  std::vector<double> signalForceWeights;
    std::vector<std::vector<double>> targetFeatures;
	double scale;
	bool usePeriodic;
  //Distances  distALg;
    HungarianAlgorithm hungAlg;

};

} // namespace PyTorchPlugin

#endif /*REFERENCE_NEURAL_NETWORK_KERNELS_H_*/
