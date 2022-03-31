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

#include "internal/PyTorchForceImpl.h"
#include "PyTorchKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <fstream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceImpl::PyTorchForceImpl(const PyTorchForce& owner) : owner(owner){
}

PyTorchForceImpl::~PyTorchForceImpl() {

}

void PyTorchForceImpl::initialize(ContextImpl& context) {


  // Gets the model name and opens it.
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    nnModule = torch::jit::load(owner.getFile());
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  kernel = context.getPlatform().createKernel(CalcPyTorchForceKernel::Name(), context);
  kernel.getAs<CalcPyTorchForceKernel>().initialize(context.getSystem(), owner, nnModule);

}

double PyTorchForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
	if ((groups&(1<<owner.getForceGroup())) != 0)
	  return kernel.getAs<CalcPyTorchForceKernel>().execute(context, includeForces, includeEnergy);
	return 0.0;
}

std::vector<std::string> PyTorchForceImpl::getKernelNames() {
	std::vector<std::string> names;
	names.push_back(CalcPyTorchForceKernel::Name());
	return names;
}
