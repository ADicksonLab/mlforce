/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NN                                    *
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

#include "PyTorchForceProxy.h"
#include "PyTorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <string>
#include <sstream>
#include <iostream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceProxy::PyTorchForceProxy() : SerializationProxy("PyTorchForce") {
}

void PyTorchForceProxy::serialize(const void* object, SerializationNode& node) const {
	node.setIntProperty("version", 1);
	const PyTorchForce& force = *reinterpret_cast<const PyTorchForce*>(object);
	node.setStringProperty("file", force.getFile());
	node.setDoubleProperty("scale", force.getScale());
	node.setIntProperty("forceGroup", force.getForceGroup());
	node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());

	std::vector<vector<double>> features = force.getTargetFeatures();
	SerializationNode& targetfeaturesNode = node.createChildNode("TargetFeatures");
	for (int i = 0; i < features.size(); i++) {
		SerializationNode&  featureNode = targetfeaturesNode.createChildNode("Particles");
		for (int j= 0; j < features[0].size(); j++){
			featureNode.createChildNode("features").setDoubleProperty("value", features[i][j]);
		}
	}

	std::vector<int>  ParticleIndices = force.getParticleIndices();
	SerializationNode& ParticleIndicesNode = node.createChildNode("ParticleIndices");

	for (int i = 0; i < ParticleIndices.size(); i++) {
		 ParticleIndicesNode.createChildNode("Index").setIntProperty("value", ParticleIndices[i]);
	}

	std::vector<double>  signalForceWeights = force.getSignalForceWeights();
	SerializationNode&  signalForceWeightsNode = node.createChildNode("SignalForceWeights");
	for (int i = 0; i < signalForceWeights.size(); i++) {
	   signalForceWeightsNode.createChildNode("Weight").setDoubleProperty("value", signalForceWeights[i]);
	}


}

void* PyTorchForceProxy::deserialize(const SerializationNode& node) const {
	if (node.getIntProperty("version") != 1)
	throw OpenMMException("Unsupported version number");

	const SerializationNode& targetfeaturesNode = node.getChildNode("TargetFeatures");
	int	numTargetParticles = targetfeaturesNode.getChildren().size();

	std::vector<std::vector<double>> targetfeatures(numTargetParticles);
	for (int i=0; i<numTargetParticles; i++){
		 const SerializationNode& featureNode = targetfeaturesNode.getChildren()[i];
		 for (auto &feature:featureNode.getChildren()){
			 targetfeatures[i].push_back(feature.getDoubleProperty("value"));

		 }
	}


	 std::vector<int> indices;
	 const SerializationNode& partilceindicesNode = node.getChildNode("ParticleIndices");
	 for (auto & index: partilceindicesNode.getChildren()) {
	   indices.push_back(index.getIntProperty("value"));
	 }

	 std::vector<double> signalForceWeights;
	 const SerializationNode& signalForceWeightsNode = node.getChildNode("SignalForceWeights");
	 for (auto &weight:signalForceWeightsNode.getChildren()){

	   signalForceWeights.push_back(weight.getDoubleProperty("value"));
	 }

	 PyTorchForce* force = new PyTorchForce(node.getStringProperty("file"),  targetfeatures,
											indices, signalForceWeights, node.getDoubleProperty("scale"));
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
