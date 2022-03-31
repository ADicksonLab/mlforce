#ifndef OPENMM_PY_TORCH_FORCE_IMPL_H_
#define OPENMM_PY_TORCH_FORCE_IMPL_H_

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
 * OTHERWISE, ARISING FROM, OUT OF OR IN COPyTorchECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "PyTorchForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <utility>
#include <set>
#include <string>

namespace PyTorchPlugin {

class System;

/**
 * This is the internal implementation of PyTorchForce.
 */

class OPENMM_EXPORT_PYTORCH PyTorchForceImpl : public OpenMM::ForceImpl {
public:
	PyTorchForceImpl(const PyTorchForce& owner);
	~PyTorchForceImpl();
	void initialize(OpenMM::ContextImpl& context);
	const PyTorchForce& getOwner() const {
	return owner;
	}
	void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
	// This force field doesn't update the state directly.
	}
	double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
	std::map<std::string, double> getDefaultParameters() {
	return std::map<std::string, double>(); // This force field doesn't define any parameters.
	}
	std::vector<std::string> getKernelNames();
private:
	const PyTorchForce& owner;
	OpenMM::Kernel kernel;
	torch::jit::script::Module nnModule;
	std::vector<std::vector<double>> targetFeatures;
	std::vector<int> particleIndicies;
};

} // namespace PyTorchPlugin

#endif /*OPENMM_PY_TORCH_FORCE_IMPL_H_*/
