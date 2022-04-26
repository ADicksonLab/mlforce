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

/**
 * This tests the CUDA implementation of TorchMLForce.
 */

#include "PyTorchForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/CustomNonbondedForce.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPyTorchCudaKernelFactories();

void testForce() {
	// Create a random cloud of particles.

	const int numParticles = 10;
	System system;
	vector<Vec3> positions(numParticles);
	OpenMM_SFMT::SFMT sfmt;
	init_gen_rand(0, sfmt);
	for (int i = 0; i < numParticles; i++) {
		system.addParticle(1.0);
		positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
	}
	std::vector<vector<double>> features(2, std::vector<double>(180));
	std::vector<int> pindices={0, 1};
	std::vector<double> weights={0.1,0.2};
	double scale = 10.0;
	PyTorchForce* force = new PyTorchForce("tests/ani_model_cpu.pt", features, pindices, weights, scale);
	system.addForce(force);

	CustomNonbondedForce* cnb_force = new CustomNonbondedForce("epsilon*(sigma/r)^12;sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)");
	cnb_force->addPerParticleParameter("sigma");
	cnb_force->addPerParticleParameter("epsilon");

	vector<double> param = {0.02, 2.0};
	for (int i = 0; i < numParticles; i++) {
	  cnb_force->addParticle(param);
	}
	cnb_force->addGlobalParameter("charge_g0", -1.2);
	cnb_force->addGlobalParameter("charge_g1", 0.5);
	cnb_force->addGlobalParameter("sigma_g0", 0.023);
	cnb_force->addGlobalParameter("sigma_g1", 0.1);
	cnb_force->addGlobalParameter("epsilon_g0", 0.03);
	cnb_force->addGlobalParameter("epsilon_g1", 1.5);
	cnb_force->addGlobalParameter("lambda_g0", 1);
	cnb_force->addGlobalParameter("lambda_g1", 1);
	cnb_force->addGlobalParameter("assignment_g0", 0);
	cnb_force->addGlobalParameter("assignment_g1", 1);

	system.addForce(cnb_force);

	// Compute the forces and energy.

	VerletIntegrator integ(1.0);
	Platform& platform = Platform::getPlatformByName("CUDA");
	Context context(system, integ, platform);
	context.setPositions(positions);
	State state = context.getState(State::Energy | State::Forces);

}


int main(int argc, char* argv[]) {
	try {
		registerPyTorchCudaKernelFactories();
		if (argc > 1)
			Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", string(argv[1]));
		testForce();
	}
	catch(const std::exception& e) {
		std::cout << "exception: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
