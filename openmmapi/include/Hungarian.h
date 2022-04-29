#ifndef HUNGARIAN_H
#define HUNGARIAN_H

/* -------------------------------------------------------------------------- *
 *                                  Hungarian                                 *
 * -------------------------------------------------------------------------- *
 * Hungarian.cpp: Header file for Class HungarianAlgorithm.					  *
 * This is a C++ wrapper with slight modification of a hungarian algorithm    *
 * implementation by Markus Buehren.										  *
 * The original implementation is a few mex-functions for use in MATLAB,      *
 * found here:																  *
 * http://www.mathworks.com/matlabcentral/fileexchange/                       *
 * 6543-functions-for-the-rectangular-assignment-problem					  *
 * Both this code and the orignal code are published under the BSD license.   *
 * by Cong Ma, 2016															  *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <vector>

using namespace std;

/**
 * This class implements the Hungarian algorithm, a solution to the Assignment Problem.
 * It finds the optimal mapping between two sets of points by minimizing the cost (distances).
 *
 */

class HungarianAlgorithm
{
public:
	/**
	 * Construct a new Hungarian Algorithm object
	 *
	 */
	HungarianAlgorithm();
	/**
	 * Destroy the Hungarian Algorithm object
	 *
	 */
	~HungarianAlgorithm();
	/**
	 * A single function wrapper for solving assignment problem.
	 * Solves the assignment problem between points that are associated with the rows
	 * and points that are associated with columns of the distance matrix
	 *
	 * @param The distance matrix
	 * @return The optimal assignment between the two sets of points
	 */
	vector<int> Solve(vector<vector<double> >& distMatrix);

private:
	/**
	 * Solve optimal solution for assignment problem using Munkres algorithm, also known as Hungarian Algorithm.
	 *
	 * @param assignment
	 * @param cost
	 * @param distMatrix
	 * @param nOfRows
	 * @param nOfColumns
	 */
	void assignmentoptimal(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns);
	void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
	void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);
	void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
	void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
};


#endif
