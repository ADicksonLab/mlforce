#ifndef OPENMM_PYTORCH_FORCE_H_
#define OPENMM_PYTORCH_FORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include "internal/windowsExportPyTorch.h"
#include<vector>

namespace PyTorchPlugin {

/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the PyTorch library to perform the computations. */

class OPENMM_EXPORT_PYTORCH PyTorchForce : public OpenMM::Force {
public:
	/**
	* Create a PyTorchForce. The Neural Network (TorchANI) model is defined by a PyTorch
	* ScriptModule and saved in the '.pt' format.
	*
	* @param file The path to the '.pt' file that contains the PyTorch model.
	* @param targetFeatures The faetures of a target ligand.
	* @param particleIndices  The Ghost partcile indices.
	* @param signalForceWeights The weight values for the dynamical variables (Signals).
	* @param scale The scale value of the force.
	*/
	PyTorchForce(const std::string& file, std::vector<std::vector<double>> targetFeatures,
				   std::vector<int> particleIndices, std::vector<double> signalForceWeights, double scale);
	/**
	* Get the path to the '.pt' file containg the TorchANI model
	*
	* @return The path to the model
	*/
	const std::string& getFile() const;
	/**
	* Get the force scale value
	*
	* @return The scale value
	*/
	const double getScale() const;
	/**
	* @brief Get the features of the target ligand
	*
	* @return The target features
	*/
	const std::vector<std::vector<double>> getTargetFeatures() const;
	/**
	* Get the Ghost partciles indices
	*
	* @return the Ghost partciles indices
	*/
	const std::vector<int> getParticleIndices() const;

	/**
	* Get the dynamical variables weights
	*
	* @return the dynamical variables weights
	*/
	const std::vector<double> getSignalForceWeights() const;
	/**
	* Set whether this force makes use of periodic boundary conditions.  If this is set
	* to true, the network must take a 3x3 tensor as its input, which
	* is set to the current periodic box vectors.
	*
	* @param The boolean value to determine if the model considers the periodic boundary conditions
	*/
	void setUsesPeriodicBoundaryConditions(bool periodic);
	/**
	* Get a boolean value showing whether the force uses the periodic boundary conditions
	*
	* @return a boolean value determining the use of periodic boundary conditions
	*/
	bool usesPeriodicBoundaryConditions() const;
	/**
	* Get the number of global parameters that the interaction depends on
	*
	* @return the number of global parameters
	*/
	int getNumGlobalParameters() const;
	/**
	 * Add a new global parameter that the interaction may depend on.  The default value provided to
	 * this method is the initial value of the parameter in newly created Contexts.  You can change
	 * the value at any time by calling setParameter() on the Context.
	 *
	 * @param name             the name of the parameter
	 * @param defaultValue     the default value of the parameter
	 * @return the index of the parameter that was added
	 */
	int addGlobalParameter(const std::string& name, double defaultValue);
	/**
	 * Get the name of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the name
	 * @return the parameter name
	 */
	const std::string& getGlobalParameterName(int index) const;
	/**
	 * Set the name of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the name
	 * @param name           the name of the parameter
	 */
	void setGlobalParameterName(int index, const std::string& name);
	/**
	 * Get the default value of a global parameter.
	 *
	 * @param index     the index of the parameter for which to get the default value
	 * @return the parameter default value
	 */
	double getGlobalParameterDefaultValue(int index) const;
	/**
	 * Set the default value of a global parameter.
	 *
	 * @param index          the index of the parameter for which to set the default value
	 * @param defaultValue   the default value of the parameter
	 */
	void setGlobalParameterDefaultValue(int index, double defaultValue);
protected:
	OpenMM::ForceImpl* createImpl() const;
private:
	class GlobalParameterInfo;
	std::string file;
	std::vector<std::vector<double>> targetFeatures;
	std::vector<int> particleIndices;
	std::vector<double> signalForceWeights;
	double scale;
	bool usePeriodic;
	std::vector<GlobalParameterInfo> globalParameters;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class PyTorchForce::GlobalParameterInfo {
public:
	std::string name;
	double defaultValue;
	GlobalParameterInfo() {
	}
	GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
	}
};

} // namespace PyTorchPlugin

#endif /*OPENMM_PYTORCHFORCE_H_*/
