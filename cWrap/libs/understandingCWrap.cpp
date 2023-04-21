#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>
#include <string.h>
#include <sstream>
#include <cstddef>
#include <sys/stat.h>
#include <chrono>
#include <bitset>

using namespace std;


void understandingCWrap_API01(float *im1_in, float *im2_in, float *multiChannelRes_inOut, int dim3, int dim2, int dim1)
{
    int sz1 = dim3 * dim2 * dim1;
    float *u1 = new float[sz1];
    float *v1 = new float[sz1];
    float *w1 = new float[sz1];
    //Copy im1_in in channel u1
    for (int i = 0; i < sz1; i++)
    {
        u1[i] = im1_in[i];
    }
    //Copy im2_in in channel v1
    for (int i = 0; i < sz1; i++)
    {
        v1[i] = im2_in[i];
    }    
    //Copy 2*im1_in - im2_in in channel w1
    for (int i = 0; i < sz1; i++)
    {
        w1[i] = 2.0*im1_in[i] - im2_in[i];
    } 
    for (int i = 0; i < sz1; i++)
    {
        multiChannelRes_inOut[i] = u1[i];
        multiChannelRes_inOut[i + sz1] = v1[i];
        multiChannelRes_inOut[i + sz1 * 2] = w1[i];
        // flow[i+sz1*3]=u1i[i]; flow[i+sz1*4]=v1i[i]; flow[i+sz1*5]=w1i[i];
    }
    delete u1;
    delete v1;
    delete w1;
}

// // some global variables
// int RAND_SAMPLES; // will all be set later (if needed)
// int image_m;
// int image_n;
// int image_o;
// int image_d = 1;
// float SSD0 = 0.0;
// float SSD1 = 0.0;
// float beta = 1;
// // float SIGMA=8.0;
// int qc = 1;

// // struct for multi-threading of mind-calculation
// struct mind_data
// {
//     float *im1;
//     float *d1;
//     uint64_t *mindq;
//     int qs;
//     int ind_d1;
// };

// /////////////////////// Contents of "transformations.h" ///////////////////
// /* several functions to interpolate and symmetrise deformations
//  calculates Jacobian and harmonic Energy */

// void interp3(float *interp, float *input, float *x1, float *y1, float *z1, int m, int n, int o, int m2, int n2, int o2, bool flag)
// {
// 	for (int k = 0; k < o; k++)
// 	{
// 		for (int j = 0; j < n; j++)
// 		{
// 			for (int i = 0; i < m; i++)
// 			{
// 				int x = floor(x1[i + j * m + k * m * n]);
// 				int y = floor(y1[i + j * m + k * m * n]);
// 				int z = floor(z1[i + j * m + k * m * n]);
// 				float dx = x1[i + j * m + k * m * n] - x;
// 				float dy = y1[i + j * m + k * m * n] - y;
// 				float dz = z1[i + j * m + k * m * n] - z;

// 				if (flag)
// 				{
// 					x += j;
// 					y += i;
// 					z += k;
// 				}
// 				interp[i + j * m + k * m * n] = (1.0 - dx) * (1.0 - dy) * (1.0 - dz) * input[min(max(y, 0), m2 - 1) + min(max(x, 0), n2 - 1) * m2 + min(max(z, 0), o2 - 1) * m2 * n2] +
// 												(1.0 - dx) * dy * (1.0 - dz) * input[min(max(y + 1, 0), m2 - 1) + min(max(x, 0), n2 - 1) * m2 + min(max(z, 0), o2 - 1) * m2 * n2] +
// 												dx * (1.0 - dy) * (1.0 - dz) * input[min(max(y, 0), m2 - 1) + min(max(x + 1, 0), n2 - 1) * m2 + min(max(z, 0), o2 - 1) * m2 * n2] +
// 												(1.0 - dx) * (1.0 - dy) * dz * input[min(max(y, 0), m2 - 1) + min(max(x, 0), n2 - 1) * m2 + min(max(z + 1, 0), o2 - 1) * m2 * n2] +
// 												dx * dy * (1.0 - dz) * input[min(max(y + 1, 0), m2 - 1) + min(max(x + 1, 0), n2 - 1) * m2 + min(max(z, 0), o2 - 1) * m2 * n2] +
// 												(1.0 - dx) * dy * dz * input[min(max(y + 1, 0), m2 - 1) + min(max(x, 0), n2 - 1) * m2 + min(max(z + 1, 0), o2 - 1) * m2 * n2] +
// 												dx * (1.0 - dy) * dz * input[min(max(y, 0), m2 - 1) + min(max(x + 1, 0), n2 - 1) * m2 + min(max(z + 1, 0), o2 - 1) * m2 * n2] +
// 												dx * dy * dz * input[min(max(y + 1, 0), m2 - 1) + min(max(x + 1, 0), n2 - 1) * m2 + min(max(z + 1, 0), o2 - 1) * m2 * n2];
// 			}
// 		}
// 	}
// }

// void filter1(float *imagein, float *imageout, int m, int n, int o, float *filter, int length, int dim)
// {
// 	int i, j, k, f;
// 	int i1, j1, k1;
// 	int hw = (length - 1) / 2;

// 	for (i = 0; i < (m * n * o); i++)
// 	{
// 		imageout[i] = 0.0;
// 	}

// 	for (k = 0; k < o; k++)
// 	{
// 		for (j = 0; j < n; j++)
// 		{
// 			for (i = 0; i < m; i++)
// 			{
// 				for (f = 0; f < length; f++)
// 				{
// 					// replicate-padding
// 					if (dim == 1)
// 						imageout[i + j * m + k * m * n] += filter[f] * imagein[max(min(i + f - hw, m - 1), 0) + j * m + k * m * n];
// 					if (dim == 2)
// 						imageout[i + j * m + k * m * n] += filter[f] * imagein[i + max(min(j + f - hw, n - 1), 0) * m + k * m * n];
// 					if (dim == 3)
// 						imageout[i + j * m + k * m * n] += filter[f] * imagein[i + j * m + max(min(k + f - hw, o - 1), 0) * m * n];
// 				}
// 			}
// 		}
// 	}
// }

// void volfilter(float *imagein, int m, int n, int o, int length, float sigma)
// {

// 	int hw = (length - 1) / 2;
// 	int i, j, f;
// 	float hsum = 0;
// 	float *filter = new float[length];
// 	for (i = 0; i < length; i++)
// 	{
// 		filter[i] = exp(-pow((i - hw), 2) / (2 * pow(sigma, 2)));
// 		hsum = hsum + filter[i];
// 	}
// 	for (i = 0; i < length; i++)
// 	{
// 		filter[i] = filter[i] / hsum;
// 	}
// 	float *image1 = new float[m * n * o];
// 	for (i = 0; i < m * n * o; i++)
// 	{
// 		image1[i] = imagein[i];
// 	}
// 	filter1(image1, imagein, m, n, o, filter, length, 1);
// 	filter1(imagein, image1, m, n, o, filter, length, 2);
// 	filter1(image1, imagein, m, n, o, filter, length, 3);

// 	delete image1;
// 	delete filter;
// }

// float jacobian(float *u1, float *v1, float *w1, int m, int n, int o, int factor)
// {

// 	float factor1 = 1.0 / (float)factor;
// 	float jmean = 0.0;
// 	float jstd = 0.0;
// 	int i;
// 	float grad[3] = {-0.5, 0.0, 0.5};
// 	float *Jac = new float[m * n * o];

// 	float *J11 = new float[m * n * o];
// 	float *J12 = new float[m * n * o];
// 	float *J13 = new float[m * n * o];
// 	float *J21 = new float[m * n * o];
// 	float *J22 = new float[m * n * o];
// 	float *J23 = new float[m * n * o];
// 	float *J31 = new float[m * n * o];
// 	float *J32 = new float[m * n * o];
// 	float *J33 = new float[m * n * o];

// 	for (i = 0; i < (m * n * o); i++)
// 	{
// 		J11[i] = 0.0;
// 		J12[i] = 0.0;
// 		J13[i] = 0.0;
// 		J21[i] = 0.0;
// 		J22[i] = 0.0;
// 		J23[i] = 0.0;
// 		J31[i] = 0.0;
// 		J32[i] = 0.0;
// 		J33[i] = 0.0;
// 	}

// 	float neg = 0;
// 	float Jmin = 1;
// 	float Jmax = 1;
// 	float J;
// 	float count = 0;
// 	float frac;

// 	filter1(u1, J11, m, n, o, grad, 3, 2);
// 	filter1(u1, J12, m, n, o, grad, 3, 1);
// 	filter1(u1, J13, m, n, o, grad, 3, 3);

// 	filter1(v1, J21, m, n, o, grad, 3, 2);
// 	filter1(v1, J22, m, n, o, grad, 3, 1);
// 	filter1(v1, J23, m, n, o, grad, 3, 3);

// 	filter1(w1, J31, m, n, o, grad, 3, 2);
// 	filter1(w1, J32, m, n, o, grad, 3, 1);
// 	filter1(w1, J33, m, n, o, grad, 3, 3);

// 	for (i = 0; i < (m * n * o); i++)
// 	{
// 		J11[i] *= factor1;
// 		J12[i] *= factor1;
// 		J13[i] *= factor1;
// 		J21[i] *= factor1;
// 		J22[i] *= factor1;
// 		J23[i] *= factor1;
// 		J31[i] *= factor1;
// 		J32[i] *= factor1;
// 		J33[i] *= factor1;
// 	}

// 	for (i = 0; i < (m * n * o); i++)
// 	{
// 		J11[i] += 1.0;
// 		J22[i] += 1.0;
// 		J33[i] += 1.0;
// 	}
// 	for (i = 0; i < (m * n * o); i++)
// 	{
// 		J = J11[i] * (J22[i] * J33[i] - J23[i] * J32[i]) - J21[i] * (J12[i] * J33[i] - J13[i] * J32[i]) + J31[i] * (J12[i] * J23[i] - J13[i] * J22[i]);
// 		jmean += J;
// 		if (J > Jmax)
// 			Jmax = J;
// 		if (J < Jmin)
// 			Jmin = J;
// 		if (J < 0)
// 			neg++;
// 		count++;
// 		Jac[i] = J;
// 	}
// 	jmean /= (m * n * o);
// 	for (int i = 0; i < m * n * o; i++)
// 	{
// 		jstd += pow(Jac[i] - jmean, 2.0);
// 	}
// 	jstd /= (m * n * o - 1);
// 	jstd = sqrt(jstd);
// 	frac = neg / count;
// 	cout << "std(J)=" << round(jstd * 100) / 100.0;
// 	// cout<<"Range: ["<<Jmin<<", "<<Jmax<<"]round(jmean*100)/100.0<<
// 	cout << " (J<0)=" << round(frac * 1e7) / 100.0 << "e-7  ";
// 	delete[] Jac;

// 	delete[] J11;
// 	delete[] J12;
// 	delete[] J13;
// 	delete[] J21;
// 	delete[] J22;
// 	delete[] J23;
// 	delete[] J31;
// 	delete[] J32;
// 	delete[] J33;

// 	return jstd;
// }

// void consistentMappingCL(float *u, float *v, float *w, float *u2, float *v2, float *w2, int m, int n, int o, int factor)
// {
// 	float factor1 = 1.0 / (float)factor;
// 	float *us = new float[m * n * o];
// 	float *vs = new float[m * n * o];
// 	float *ws = new float[m * n * o];
// 	float *us2 = new float[m * n * o];
// 	float *vs2 = new float[m * n * o];
// 	float *ws2 = new float[m * n * o];

// 	for (int i = 0; i < m * n * o; i++)
// 	{
// 		us[i] = u[i] * factor1;
// 		vs[i] = v[i] * factor1;
// 		ws[i] = w[i] * factor1;
// 		us2[i] = u2[i] * factor1;
// 		vs2[i] = v2[i] * factor1;
// 		ws2[i] = w2[i] * factor1;
// 	}

// 	for (int it = 0; it < 10; it++)
// 	{
// 		interp3(u, us2, us, vs, ws, m, n, o, m, n, o, true);
// 		interp3(v, vs2, us, vs, ws, m, n, o, m, n, o, true);
// 		interp3(w, ws2, us, vs, ws, m, n, o, m, n, o, true);
// 		for (int i = 0; i < m * n * o; i++)
// 		{
// 			u[i] = 0.5 * us[i] - 0.5 * u[i];
// 			v[i] = 0.5 * vs[i] - 0.5 * v[i];
// 			w[i] = 0.5 * ws[i] - 0.5 * w[i];
// 		}
// 		interp3(u2, us, us2, vs2, ws2, m, n, o, m, n, o, true);
// 		interp3(v2, vs, us2, vs2, ws2, m, n, o, m, n, o, true);
// 		interp3(w2, ws, us2, vs2, ws2, m, n, o, m, n, o, true);
// 		for (int i = 0; i < m * n * o; i++)
// 		{
// 			u2[i] = 0.5 * us2[i] - 0.5 * u2[i];
// 			v2[i] = 0.5 * vs2[i] - 0.5 * v2[i];
// 			w2[i] = 0.5 * ws2[i] - 0.5 * w2[i];
// 		}

// 		for (int i = 0; i < m * n * o; i++)
// 		{
// 			us[i] = u[i];
// 			vs[i] = v[i];
// 			ws[i] = w[i];
// 			us2[i] = u2[i];
// 			vs2[i] = v2[i];
// 			ws2[i] = w2[i];
// 		}
// 	}

// 	for (int i = 0; i < m * n * o; i++)
// 	{
// 		u[i] *= (float)factor;
// 		v[i] *= (float)factor;
// 		w[i] *= (float)factor;
// 		u2[i] *= (float)factor;
// 		v2[i] *= (float)factor;
// 		w2[i] *= (float)factor;
// 	}

// 	delete us;
// 	delete vs;
// 	delete ws;
// 	delete us2;
// 	delete vs2;
// 	delete ws2;
// }

// void upsampleDeformationsCL(float *u1, float *v1, float *w1, float *u0, float *v0, float *w0, int m, int n, int o, int m2, int n2, int o2)
// {

// 	float scale_m = (float)m / (float)m2;
// 	float scale_n = (float)n / (float)n2;
// 	float scale_o = (float)o / (float)o2;

// 	float *x1 = new float[m * n * o];
// 	float *y1 = new float[m * n * o];
// 	float *z1 = new float[m * n * o];
// 	for (int k = 0; k < o; k++)
// 	{
// 		for (int j = 0; j < n; j++)
// 		{
// 			for (int i = 0; i < m; i++)
// 			{
// 				x1[i + j * m + k * m * n] = j / scale_n;
// 				y1[i + j * m + k * m * n] = i / scale_m;
// 				z1[i + j * m + k * m * n] = k / scale_o;
// 			}
// 		}
// 	}

// 	interp3(u1, u0, x1, y1, z1, m, n, o, m2, n2, o2, false);
// 	interp3(v1, v0, x1, y1, z1, m, n, o, m2, n2, o2, false);
// 	interp3(w1, w0, x1, y1, z1, m, n, o, m2, n2, o2, false);

// 	delete[] x1;
// 	delete[] y1;
// 	delete[] z1;

// 	// for(int i=0;i<m2*n2*o2;i++){
// 	//	u2[i]*=scale_n;
// 	//	v2[i]*=scale_m;
// 	//	w2[i]*=scale_o;
// 	// }
// }


// /////////////////////// Contents of "primsMST.h" ///////////////////
// /* Image-driven minimum-spanning-tree calcuation using Prim's algorithm.
//  Average run-time should be of n*log(n) complexity.
//  Uses heap data structure to speed-up finding the next lowest edge-weight.
//  Edge-weights
// */

// class Edge
// {
// public:
// 	double weight;
// 	int vert1;
// 	int vert2;
// 	Edge(double w = 0, int v1 = 0, int v2 = 0);
// 	bool operator<(const Edge &b) const;
// 	void print();
// };

// Edge::Edge(double w, int v1, int v2)
// {
// 	weight = w;
// 	vert1 = v1;
// 	vert2 = v2;
// }

// bool Edge::operator<(const Edge &b) const
// {
// 	return (this->weight > b.weight);
// }

// int newEdge(Edge edge1, Edge &edgeout, bool *vertices)
// {
// 	bool new1 = vertices[edge1.vert1];
// 	bool new2 = vertices[edge1.vert2];
// 	int out1;
// 	if (new1 ^ new2)
// 	{
// 		if (new1)
// 		{
// 			out1 = edge1.vert2;
// 			edgeout = Edge(edge1.weight, edge1.vert1, edge1.vert2);
// 		}
// 		else
// 		{
// 			out1 = edge1.vert1;
// 			edgeout = Edge(edge1.weight, edge1.vert2, edge1.vert1);
// 		}
// 	}
// 	else
// 	{
// 		out1 = -1;
// 	}
// 	return out1;
// }

// float edgecost2weight(float val, float meanim)
// {
// 	return exp(-val / meanim);
// }

// void primsGraph(float *im1, int *ordered, int *parents, float *edgemst, int step1, int m2, int n2, int o2)
// {

// 	int m = m2 / step1;
// 	int n = n2 / step1;
// 	int o = o2 / step1;

// 	int num_vertices = m * n * o;
// 	int sz = num_vertices;
// 	int len = m * n * o;

// 	int num_neighbours = 6;
// 	float *edgecost = new float[num_vertices * num_neighbours];
// 	int *index_neighbours = new int[num_vertices * num_neighbours];
// 	for (int i = 0; i < num_vertices * num_neighbours; i++)
// 	{
// 		edgecost[i] = 0.0;
// 		index_neighbours[i] = -1;
// 	}

// 	int dx[6] = {-1, 1, 0, 0, 0, 0};
// 	int dy[6] = {0, 0, -1, 1, 0, 0};
// 	int dz[6] = {0, 0, 0, 0, -1, 1};
// 	int xx, yy, zz, xx2, yy2, zz2;
// 	// calculate edge-weights based on SAD of groups of voxels (for each control-point)
// 	for (int k = 0; k < o; k++)
// 	{
// 		for (int j = 0; j < n; j++)
// 		{
// 			for (int i = 0; i < m; i++)
// 			{
// 				for (int nb = 0; nb < num_neighbours; nb++)
// 				{
// 					if ((i + dy[nb]) >= 0 & (i + dy[nb]) < m & (j + dx[nb]) >= 0 & (j + dx[nb]) < n & (k + dz[nb]) >= 0 & (k + dz[nb]) < o)
// 					{
// 						index_neighbours[i + j * m + k * m * n + nb * num_vertices] = i + dy[nb] + (j + dx[nb]) * m + (k + dz[nb]) * m * n;
// 						// float randv=((float)rand()/float(RAND_MAX));
// 						// edgecost[i+j*m+k*m*n+nb*num_vertices]=randv;
// 						for (int k1 = 0; k1 < step1; k1++)
// 						{
// 							for (int j1 = 0; j1 < step1; j1++)
// 							{
// 								for (int i1 = 0; i1 < step1; i1++)
// 								{
// 									xx = j * step1 + j1;
// 									yy = i * step1 + i1;
// 									zz = k * step1 + k1;
// 									xx2 = (j + dx[nb]) * step1 + j1;
// 									yy2 = (i + dy[nb]) * step1 + i1;
// 									zz2 = (k + dz[nb]) * step1 + k1;
// 									edgecost[i + j * m + k * m * n + nb * num_vertices] += fabs(im1[yy + xx * m2 + zz * m2 * n2] - im1[yy2 + xx2 * m2 + zz2 * m2 * n2]);
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	float meanim = 0.0;
// 	for (int i = 0; i < m2 * n2 * o2; i++)
// 	{
// 		meanim += im1[i];
// 	}
// 	meanim /= (float)(m2 * n2 * o2);
// 	float stdim = 0.0;
// 	for (int i = 0; i < m2 * n2 * o2; i++)
// 	{
// 		stdim += pow(im1[i] - meanim, 2);
// 	}
// 	stdim = sqrt(stdim / (float)(m2 * n2 * o2));

// 	for (int i = 0; i < sz * 6; i++)
// 	{
// 		edgecost[i] /= (float)pow(step1, 3);
// 	}
// 	for (int i = 0; i < sz * 6; i++)
// 	{
// 		edgecost[i] = -edgecost2weight(edgecost[i], 2.0f * stdim);
// 	}

// 	float centrex = n / 2;
// 	float centrey = m / 2;
// 	float centrez = o / 2;

// 	int root = m / 2 + n / 2 * m + o / 2 * m * n;

// 	vector<Edge> priority;
// 	bool *vertices = new bool[num_vertices];
// 	int *level = new int[num_vertices];
// 	for (int i = 0; i < num_vertices; i++)
// 	{
// 		vertices[i] = false;
// 		parents[i] = -1;
// 	}
// 	// int root=0;
// 	level[root] = 0;
// 	int last = root;
// 	vertices[root] = true;
// 	Edge edgeout = Edge(0.0, -1, -1);
// 	Edge minedge = Edge(0.0, -1, -1);
// 	float cost = 0.0;
// 	auto time1 = chrono::steady_clock::now();

// 	for (int i = 0; i < num_vertices - 1; i++)
// 	{ // run n-1 times to have all vertices added
// 		// add edges of new vertex to priority queue
// 		for (int j = 0; j < num_neighbours; j++)
// 		{
// 			int n = index_neighbours[last + j * num_vertices];
// 			if (n >= 0)
// 			{
// 				priority.push_back(Edge(edgecost[last + j * num_vertices], last, n));
// 				push_heap(priority.begin(), priority.end());
// 			}
// 		}
// 		last = -1;
// 		// find valid edge with lowest weight (main step of Prim's algorithm)
// 		while (last == -1)
// 		{
// 			minedge = priority.front();
// 			pop_heap(priority.begin(), priority.end());
// 			priority.pop_back();
// 			bool new1 = vertices[minedge.vert1]; // is either vertex already part of MST?
// 			bool new2 = vertices[minedge.vert2];
// 			last = newEdge(minedge, edgeout, vertices); // return next valid vertex or -1 if edge exists already
// 		}
// 		cost += edgeout.weight;
// 		vertices[last] = true;
// 		level[edgeout.vert2] = level[edgeout.vert1] + 1;
// 		parents[edgeout.vert2] = edgeout.vert1;
// 	}

// 	// find correct ordering in constant time
// 	int maxlevel = 0;
// 	for (int i = 0; i < num_vertices; i++)
// 	{
// 		if (level[i] > maxlevel)
// 			maxlevel = level[i];
// 	}
// 	maxlevel++;
// 	int *leveloffset = new int[maxlevel];
// 	int *levelcount = new int[maxlevel];
// 	for (int i = 0; i < maxlevel; i++)
// 	{
// 		leveloffset[i] = 0;
// 		levelcount[i] = 0;
// 	}
// 	for (int i = 0; i < num_vertices; i++)
// 	{
// 		if (level[i] < maxlevel - 1)
// 			leveloffset[level[i] + 1]++; // counting number of vertices in each level
// 	}
// 	for (int i = 1; i < maxlevel; i++)
// 	{
// 		leveloffset[i] += leveloffset[i - 1]; // cumulative sum
// 	}
// 	for (int i = 0; i < num_vertices; i++)
// 	{
// 		int num = leveloffset[level[i]] + levelcount[level[i]];
// 		levelcount[level[i]]++;
// 		ordered[num] = i;
// 	}

// 	auto time2 = chrono::steady_clock::now();
// 	double timeAll = chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
// 	nth_element(levelcount, levelcount + maxlevel / 2, levelcount + maxlevel);
// 	// printf("Prims algorithm with %d levels finished in %f secs.\nMaximum %d, minimum %d, mean %d, and median %d width of tree.\n",
// 	//  maxlevel,timeAll,*max_element(levelcount,levelcount+maxlevel),*min_element(levelcount,levelcount+maxlevel),(int)(num_vertices/maxlevel),levelcount[maxlevel/2]);
// 	for (int i = 0; i < sz; i++)
// 	{
// 		edgemst[i] = 0.0f;
// 	}
// 	for (int i = 1; i < sz; i++)
// 	{
// 		int ochild = ordered[i];
// 		int oparent = parents[ordered[i]];
// 		for (int nb = 0; nb < num_neighbours; nb++)
// 		{
// 			int z = ochild / (m * n);
// 			int x = (ochild - z * m * n) / m;
// 			int y = ochild - z * m * n - x * m;
// 			int index = y + dy[nb] + (x + dx[nb]) * m + (z + dz[nb]) * m * n;
// 			if (index == oparent)
// 			{
// 				edgemst[ochild] = -edgecost[ochild + nb * sz];
// 			}
// 		}
// 	}
// 	priority.clear();

// 	delete edgecost;
// 	delete index_neighbours;
// 	delete levelcount;
// 	delete leveloffset;
// 	delete vertices;
// 	delete level;
// }


// /////////////////////// Contents of "regularisation.h" ///////////////////
// /* Incremental diffusion regularisation of parametrised transformation
//  using (globally optimal) belief-propagation on minimum spanning tree.
//  Fast distance transform uses squared differences.
//  Similarity cost for each node and label has to be given as input.
// */
// void messageDT(int ind, float *data, short *indout, int len1, float offsetx, float offsety, float offsetz)
// {

//     // int ind1=get_global_id(0)+start;
//     //    int ind=ordered[ind1];

//     int len2 = len1 * len1;
//     int len3 = len1 * len1 * len1;
//     float *z = new float[len1 * 2 + 1];

//     float *val;
//     float *valout;
//     short *indo;

//     float *valb;
//     float *valb2;

//     float *buffer = new float[len3];
//     float *buffer2 = new float[len3];
//     int *indb;
//     int *indb2;
//     int *bufferi = new int[len3];
//     int *bufferi2 = new int[len3];

//     for (int i = 0; i < len1 * 2 + 1; i++)
//     {
//         z[i] = (i - len1 + offsety) * (i - len1 + offsety);
//     }

//     for (int k1 = 0; k1 < len1; k1++)
//     {
//         for (int j1 = 0; j1 < len1; j1++)
//         {
//             // valb=buffer2+(j1*len1+k1*len1*len1);//
//             val = data + ind * len3 + (j1 * len1 + k1 * len1 * len1);
//             valb2 = buffer + (j1 * len1 + k1 * len1 * len1);
//             indb = bufferi + (j1 * len1 + k1 * len1 * len1);
//             int num = (j1 * len1 + k1 * len1 * len1);
//             for (int i = 0; i < len1; i++)
//             {
//                 float minval = val[0] + z[i + len1];
//                 int minind = 0;
//                 for (int j = 0; j < len1; j++)
//                 {
//                     bool b = (val[j] + z[i - j + len1] < minval);
//                     minval = b ? val[j] + z[i - j + len1] : minval;
//                     minind = b ? j : minind;
//                 }
//                 valb2[i] = minval;
//                 indb[i] = minind + num;
//             }
//         }
//     }
//     for (int i = 0; i < len1 * 2; i++)
//     {
//         z[i] = (i - len1 + offsetx) * (i - len1 + offsetx);
//     }
//     for (int k1 = 0; k1 < len1; k1++)
//     {
//         for (int i1 = 0; i1 < len1; i1++)
//         {
//             valb = buffer + (i1 + k1 * len1 * len1);
//             valb2 = buffer2 + (i1 + k1 * len1 * len1);
//             indb = bufferi + (i1 + k1 * len1 * len1);
//             indb2 = bufferi2 + (i1 + k1 * len1 * len1);
//             for (int i = 0; i < len1; i++)
//             {
//                 float minval = valb[0] + z[i + len1];
//                 int minind = 0;
//                 for (int j = 0; j < len1; j++)
//                 {
//                     bool b = (valb[j * len1] + z[i - j + len1] < minval);
//                     minval = b ? valb[j * len1] + z[i - j + len1] : minval;
//                     minind = b ? j : minind;
//                 }
//                 valb2[i * len1] = minval;
//                 indb2[i * len1] = indb[minind * len1];
//             }
//         }
//     }
//     for (int i = 0; i < len1 * 2; i++)
//     {
//         z[i] = (i - len1 + offsetz) * (i - len1 + offsetz);
//     }
//     for (int j1 = 0; j1 < len1; j1++)
//     {
//         for (int i1 = 0; i1 < len1; i1++)
//         {
//             valb = buffer2 + (i1 + j1 * len1);
//             // valb2=buffer+(i1+j1*len1);
//             valout = data + ind * len3 + (i1 + j1 * len1);
//             indb = bufferi2 + (i1 + j1 * len1);
//             // indb2=bufferi+(i1+j1*len1);
//             indo = indout + ind * len3 + (i1 + j1 * len1);
//             for (int i = 0; i < len1; i++)
//             {
//                 float minval = valb[0] + z[i + len1];
//                 int minind = 0;
//                 for (int j = 0; j < len1; j++)
//                 {
//                     bool b = (valb[j * len2] + z[i - j + len1] < minval);
//                     minval = b ? valb[j * len2] + z[i - j + len1] : minval;
//                     minind = b ? j : minind;
//                 }
//                 valout[i * len2] = minval;
//                 indo[i * len2] = indb[minind * len2];
//             }
//         }
//     }
//     delete z;
//     delete buffer;
//     delete buffer2;
//     delete bufferi;
//     delete bufferi2;
// }

// void regularisationCL(float *costall, float *u0, float *v0, float *w0, float *u1, float *v1, float *w1, int hw, int step1, float quant, int *ordered, int *parents, float *edgemst)
// {

//     int m2 = image_m;
//     int n2 = image_n;
//     int o2 = image_o;

//     int m = m2 / step1;
//     int n = n2 / step1;
//     int o = o2 / step1;

//     int sz = m * n * o;
//     int len = hw * 2 + 1;
//     int len1 = len;
//     int len2 = len * len * len;
//     int len3 = len * len * len;

//     auto time1 = chrono::steady_clock::now();

//     short *allinds = new short[sz * len2];
//     float *cost1 = new float[len2];
//     float *vals = new float[len2];
//     int *inds = new int[len2];

//     // calculate level boundaries for parallel implementation
//     int *levels = new int[sz];
//     for (int i = 0; i < sz; i++)
//     {
//         levels[i] = 0;
//     }
//     for (int i = 1; i < sz; i++)
//     {
//         int ochild = ordered[i];
//         int oparent = parents[ordered[i]];
//         levels[ochild] = levels[oparent] + 1;
//     }
//     int maxlev = 1 + *max_element(levels, levels + sz);

//     int *numlev = new int[maxlev];

//     int *startlev = new int[maxlev];
//     for (int i = 0; i < maxlev; i++)
//     {
//         numlev[i] = 0;
//     }
//     for (int i = 0; i < sz; i++)
//     {
//         numlev[levels[i]]++;
//     }
//     startlev[0] = numlev[0];
//     for (int i = 1; i < maxlev; i++)
//     { // cumulative sum
//         startlev[i] = startlev[i - 1] + numlev[i];
//     }
//     delete levels;

//     int xs1, ys1, zs1, xx, yy, zz, xx2, yy2, zz2;

//     for (int i = 0; i < len2; i++)
//     {
//         cost1[i] = 0;
//     }

//     // MAIN LOOP - TO BE PARALLELISED
//     int frac = (int)(sz / 25);
//     int counti = 0;
//     int counti2 = 0;

//     bool *processed = new bool[sz];
//     for (int i = 0; i < sz; i++)
//     {
//         processed[i] = false;
//     }
//     int dblcount = 0;
//     float timeCopy = 0;
//     float timeMessage = 0;
//     // calculate mst-cost
//     for (int lev = maxlev - 1; lev > 0; lev--)
//     {
//         int start = startlev[lev - 1];
//         int length = numlev[lev];

//         time1 = chrono::steady_clock::now();

//         for (int i = start; i < start + length; i++)
//         {
//             int ochild = ordered[i];
//             for (int l = 0; l < len2; l++)
//             {
//                 costall[ochild * len2 + l] *= edgemst[ochild];
//             }
//         }
// #pragma omp parallel for
//         for (int i = start; i < start + length; i++)
//         {
//             int ochild = ordered[i];
//             int oparent = parents[ordered[i]];

//             float offsetx = (u0[oparent] - u0[ochild]) / (float)quant;
//             float offsety = (v0[oparent] - v0[ochild]) / (float)quant;
//             float offsetz = (w0[oparent] - w0[ochild]) / (float)quant;
//             messageDT(ochild, costall, allinds, len1, offsetx, offsety, offsetz);
//         }

//         auto time2 = chrono::steady_clock::now();
//         timeMessage += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();

//         time1 = chrono::steady_clock::now();

//         // copy necessary if vectorisation is used (otherwise multiple simultaneous +='s)
//         int start0 = startlev[lev - 1];
//         int length0 = numlev[lev];
//         for (int i = start0; i < start0 + length0; i++)
//         {
//             int ochild = ordered[i];
//             int oparent = parents[ordered[i]];
//             float minval = *min_element(costall + ochild * len2, costall + ochild * len2 + len3);
//             for (int l = 0; l < len2; l++)
//             {
//                 costall[oparent * len2 + l] += (costall[ochild * len2 + l] - minval); /// edgemst[ochild];//transp
//                 // edgemst[ochild]*
//             }
//         }

//         time2 = chrono::steady_clock::now();
//         timeCopy += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//     }

//     // dense displacement space
//     float *xs = new float[len * len * len];
//     float *ys = new float[len * len * len];
//     float *zs = new float[len * len * len];

//     for (int i = 0; i < len; i++)
//     {
//         for (int j = 0; j < len; j++)
//         {
//             for (int k = 0; k < len; k++)
//             {
//                 xs[i + j * len + k * len * len] = (j - hw) * quant;
//                 ys[i + j * len + k * len * len] = (i - hw) * quant;
//                 zs[i + j * len + k * len * len] = (k - hw) * quant;
//             }
//         }
//     }

//     int *selected = new int[sz];

//     // mst-cost & select displacement for root note
//     int i = 0;
//     int oroot = ordered[i];
//     for (int l = 0; l < len2; l++)
//     {
//         cost1[l] = costall[oroot * len2 + l]; // transp
//     }
//     float value = cost1[0];
//     int index = 0;

//     for (int l = 0; l < len2; l++)
//     {
//         if (cost1[l] < value)
//         {
//             value = cost1[l];
//             index = l;
//         }
//         allinds[oroot * len2 + l] = l; // transp
//     }
//     selected[oroot] = index;
//     u1[oroot] = xs[index] + u0[oroot];
//     v1[oroot] = ys[index] + v0[oroot];
//     w1[oroot] = zs[index] + w0[oroot];

//     // select displacements and add to previous deformation field
//     for (int i = 1; i < sz; i++)
//     {
//         int ochild = ordered[i];
//         int oparent = parents[ordered[i]];
//         // select from argmin of based on parent selection
//         // index=allinds[ochild+selected[oparent]*sz];
//         index = allinds[ochild * len2 + selected[oparent]]; // transp
//         selected[ochild] = index;
//         u1[ochild] = xs[index] + u0[ochild];
//         v1[ochild] = ys[index] + v0[ochild];
//         w1[ochild] = zs[index] + w0[ochild];
//     }

//     // cout<<"Deformation field calculated!\n";

//     delete cost1;
//     delete vals;
//     delete inds;
//     delete allinds;
//     delete selected;
// }


// /////////////////////// Contents of "MINDSSCbox.h" ///////////////////
// void boxfilter(float *input, float *temp1, float *temp2, int hw, int m, int n, int o)
// {

//     int sz = m * n * o;
//     for (int i = 0; i < sz; i++)
//     {
//         temp1[i] = input[i];
//     }

//     for (int k = 0; k < o; k++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 1; i < m; i++)
//             {
//                 temp1[i + j * m + k * m * n] += temp1[(i - 1) + j * m + k * m * n];
//             }
//         }
//     }

//     for (int k = 0; k < o; k++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 0; i < (hw + 1); i++)
//             {
//                 temp2[i + j * m + k * m * n] = temp1[(i + hw) + j * m + k * m * n];
//             }
//             for (int i = (hw + 1); i < (m - hw); i++)
//             {
//                 temp2[i + j * m + k * m * n] = temp1[(i + hw) + j * m + k * m * n] - temp1[(i - hw - 1) + j * m + k * m * n];
//             }
//             for (int i = (m - hw); i < m; i++)
//             {
//                 temp2[i + j * m + k * m * n] = temp1[(m - 1) + j * m + k * m * n] - temp1[(i - hw - 1) + j * m + k * m * n];
//             }
//         }
//     }

//     for (int k = 0; k < o; k++)
//     {
//         for (int j = 1; j < n; j++)
//         {
//             for (int i = 0; i < m; i++)
//             {
//                 temp2[i + j * m + k * m * n] += temp2[i + (j - 1) * m + k * m * n];
//             }
//         }
//     }

//     for (int k = 0; k < o; k++)
//     {
//         for (int i = 0; i < m; i++)
//         {
//             for (int j = 0; j < (hw + 1); j++)
//             {
//                 temp1[i + j * m + k * m * n] = temp2[i + (j + hw) * m + k * m * n];
//             }
//             for (int j = (hw + 1); j < (n - hw); j++)
//             {
//                 temp1[i + j * m + k * m * n] = temp2[i + (j + hw) * m + k * m * n] - temp2[i + (j - hw - 1) * m + k * m * n];
//             }
//             for (int j = (n - hw); j < n; j++)
//             {
//                 temp1[i + j * m + k * m * n] = temp2[i + (n - 1) * m + k * m * n] - temp2[i + (j - hw - 1) * m + k * m * n];
//             }
//         }
//     }

//     for (int k = 1; k < o; k++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 0; i < m; i++)
//             {
//                 temp1[i + j * m + k * m * n] += temp1[i + j * m + (k - 1) * m * n];
//             }
//         }
//     }

//     for (int j = 0; j < n; j++)
//     {
//         for (int i = 0; i < m; i++)
//         {
//             for (int k = 0; k < (hw + 1); k++)
//             {
//                 input[i + j * m + k * m * n] = temp1[i + j * m + (k + hw) * m * n];
//             }
//             for (int k = (hw + 1); k < (o - hw); k++)
//             {
//                 input[i + j * m + k * m * n] = temp1[i + j * m + (k + hw) * m * n] - temp1[i + j * m + (k - hw - 1) * m * n];
//             }
//             for (int k = (o - hw); k < o; k++)
//             {
//                 input[i + j * m + k * m * n] = temp1[i + j * m + (o - 1) * m * n] - temp1[i + j * m + (k - hw - 1) * m * n];
//             }
//         }
//     }
// }

// void imshift(float *input, float *output, int dx, int dy, int dz, int m, int n, int o)
// {
//     for (int k = 0; k < o; k++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 0; i < m; i++)
//             {
//                 if (i + dy >= 0 && i + dy < m && j + dx >= 0 && j + dx < n && k + dz >= 0 && k + dz < o)
//                     output[i + j * m + k * m * n] = input[i + dy + (j + dx) * m + (k + dz) * m * n];
//                 else
//                     output[i + j * m + k * m * n] = input[i + j * m + k * m * n];
//             }
//         }
//     }
// }

// /*void *distances(void *threadarg)
// {
//     struct mind_data *my_data;
//     my_data = (struct mind_data *) threadarg;
//     float* im1=my_data->im1;
//     float* d1=my_data->d1;
//     int qs=my_data->qs;
//     int ind_d1=my_data->ind_d1;
//     int m=image_m;
//     int n=image_n;
//     int o=image_o;*/

// void distances(float *im1, float *d1, int m, int n, int o, int qs, int l)
// {
//     int sz1 = m * n * o;
//     float *w1 = new float[sz1];
//     int len1 = 6;

//     float *temp1 = new float[sz1];
//     float *temp2 = new float[sz1];
//     int dx[6] = {+qs, +qs, -qs, +0, +qs, +0};
//     int dy[6] = {+qs, -qs, +0, -qs, +0, +qs};
//     int dz[6] = {0, +0, +qs, +qs, +qs, +qs};

//     imshift(im1, w1, dx[l], dy[l], dz[l], m, n, o);
//     for (int i = 0; i < sz1; i++)
//     {
//         w1[i] = (w1[i] - im1[i]) * (w1[i] - im1[i]);
//     }
//     boxfilter(w1, temp1, temp2, qs, m, n, o);
//     for (int i = 0; i < sz1; i++)
//     {
//         d1[i + l * sz1] = w1[i];
//     }

//     delete temp1;
//     delete temp2;
//     delete w1;
// }

// //__builtin_popcountll(left[i]^right[i]); absolute hamming distances
// void descriptor(uint64_t *mindq, float *im1, int m, int n, int o, int qs)
// {
//     // MIND with self-similarity context

//     int dx[6] = {+qs, +qs, -qs, +0, +qs, +0};
//     int dy[6] = {+qs, -qs, +0, -qs, +0, +qs};
//     int dz[6] = {0, +0, +qs, +qs, +qs, +qs};

//     int sx[12] = {-qs, +0, -qs, +0, +0, +qs, +0, +0, +0, -qs, +0, +0};
//     int sy[12] = {+0, -qs, +0, +qs, +0, +0, +0, +qs, +0, +0, +0, -qs};
//     int sz[12] = {+0, +0, +0, +0, -qs, +0, -qs, +0, -qs, +0, -qs, +0};

//     int index[12] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

//     float sigma = 0.75; // 1.0;//0.75;//1.5;
//     int rho = ceil(sigma * 1.5) * 2 + 1;

//     int len1 = 6;
//     const int len2 = 12;

//     image_d = 12;
//     int d = 12;
//     int sz1 = m * n * o;

//     //============== DISTANCES USING BOXFILTER ===================
//     float *d1 = new float[sz1 * len1];
//     auto time1 = chrono::steady_clock::now();

// #pragma omp parallel for
//     for (int l = 0; l < len1; l++)
//     {
//         distances(im1, d1, m, n, o, qs, l);
//     }

//     auto time2 = chrono::steady_clock::now();
//     float timeMIND1 = chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//     time1 = chrono::steady_clock::now();

//     // quantisation table
//     const int val = 6;

//     const uint64_t power = 32;

// #pragma omp parallel for
//     for (int k = 0; k < o; k++)
//     {
//         unsigned int tablei[6] = {0, 1, 3, 7, 15, 31};
//         float compare[val - 1];
//         for (int i = 0; i < val - 1; i++)
//         {
//             compare[i] = -log((i + 1.5f) / val);
//         }
//         float mind1[12];
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 0; i < m; i++)
//             {
//                 for (int l = 0; l < len2; l++)
//                 {
//                     if (i + sy[l] >= 0 && i + sy[l] < m && j + sx[l] >= 0 && j + sx[l] < n && k + sz[l] >= 0 && k + sz[l] < o)
//                     {
//                         mind1[l] = d1[i + sy[l] + (j + sx[l]) * m + (k + sz[l]) * m * n + index[l] * sz1];
//                     }
//                     else
//                     {
//                         mind1[l] = d1[i + j * m + k * m * n + index[l] * sz1];
//                     }
//                 }
//                 float minval = *min_element(mind1, mind1 + len2);
//                 float sumnoise = 0.0f;
//                 for (int l = 0; l < len2; l++)
//                 {
//                     mind1[l] -= minval;
//                     sumnoise += mind1[l];
//                 }
//                 float noise1 = max(sumnoise / (float)len2, 1e-6f);
//                 for (int l = 0; l < len2; l++)
//                 {
//                     mind1[l] /= noise1;
//                 }
//                 uint64_t accum = 0;
//                 uint64_t tabled1 = 1;

//                 for (int l = 0; l < len2; l++)
//                 {
//                     // mind1[l]=exp(-mind1[l]);
//                     int mind1val = 0;
//                     for (int c = 0; c < val - 1; c++)
//                     {
//                         mind1val += compare[c] > mind1[l] ? 1 : 0;
//                     }
//                     // int mind1val=min(max((int)(mind1[l]*val-0.5f),0),val-1);
//                     accum += tablei[mind1val] * tabled1;
//                     tabled1 *= power;
//                 }
//                 mindq[i + j * m + k * m * n] = accum;
//             }
//         }
//     }

//     time2 = chrono::steady_clock::now();
//     float timeMIND2 = chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//     delete d1;
// }



// /////////////////////// Contents of "dataCostD.h" /////////////////////
// void interp3xyz(float *datai, float *data, float *datax, float *datay, int len1, int len2)
// {
//     // x-interp
//     for (int k = 0; k < len1; k++)
//     {
//         for (int j = 0; j < len2; j++)
//         {
//             int j2 = (j + 1) / 2;
//             if (j % 2 == 1)
//             {
//                 for (int i = 0; i < len1; i++)
//                 {
//                     datax[i + j * len1 + k * len1 * len2] = data[i + j2 * len1 + k * len1 * len1];
//                 }
//             }
//             else
//                 for (int i = 0; i < len1; i++)
//                 {
//                     datax[i + j * len1 + k * len1 * len2] = 0.5 * (data[i + j2 * len1 + k * len1 * len1] + data[i + (j2 + 1) * len1 + k * len1 * len1]);
//                 }
//         }
//     }

//     // y-interp
//     for (int k = 0; k < len1; k++)
//     {
//         for (int j = 0; j < len2; j++)
//         {
//             for (int i = 0; i < len2; i++)
//             {
//                 int i2 = (i + 1) / 2;
//                 if (i % 2 == 1)
//                     datay[i + j * len2 + k * len2 * len2] = datax[i2 + j * len1 + k * len1 * len2];
//                 else
//                     datay[i + j * len2 + k * len2 * len2] = 0.5 * (datax[i2 + j * len1 + k * len1 * len2] + datax[i2 + 1 + j * len1 + k * len1 * len2]);
//             }
//         }
//     }

//     // z-interp
//     for (int k = 0; k < len2; k++)
//     {
//         int k2 = (k + 1) / 2;
//         if (k % 2 == 1)
//         {
//             for (int j = 0; j < len2; j++)
//             {
//                 for (int i = 0; i < len2; i++)
//                 {
//                     datai[i + j * len2 + k * len2 * len2] = datay[i + j * len2 + k2 * len2 * len2];
//                 }
//             }
//         }
//         else
//         {
//             for (int j = 0; j < len2; j++)
//             {
//                 for (int i = 0; i < len2; i++)
//                 {
//                     datai[i + j * len2 + k * len2 * len2] = 0.5 * (datay[i + j * len2 + k2 * len2 * len2] + datay[i + j * len2 + (k2 + 1) * len2 * len2]);
//                 }
//             }
//         }
//     }
// }

// void interp3xyzB(float *datai, float *data, float *datax, float *datay, int len1, int len2)
// {
//     // x-interp
//     for (int k = 0; k < len1; k++)
//     {
//         for (int j = 0; j < len2; j++)
//         {
//             int j2 = (j + 1) / 2;
//             if (j % 2 == 0)
//             {
//                 for (int i = 0; i < len1; i++)
//                 {
//                     datax[i + j * len1 + k * len1 * len2] = data[i + j2 * len1 + k * len1 * len1];
//                 }
//             }
//             else
//                 for (int i = 0; i < len1; i++)
//                 {
//                     datax[i + j * len1 + k * len1 * len2] = 0.5 * (data[i + j2 * len1 + k * len1 * len1] + data[i + (j2 - 1) * len1 + k * len1 * len1]);
//                 }
//         }
//     }

//     // y-interp
//     for (int k = 0; k < len1; k++)
//     {
//         for (int j = 0; j < len2; j++)
//         {
//             for (int i = 0; i < len2; i++)
//             {
//                 int i2 = (i + 1) / 2;
//                 if (i % 2 == 0)
//                     datay[i + j * len2 + k * len2 * len2] = datax[i2 + j * len1 + k * len1 * len2];
//                 else
//                     datay[i + j * len2 + k * len2 * len2] = 0.5 * (datax[i2 + j * len1 + k * len1 * len2] + datax[i2 - 1 + j * len1 + k * len1 * len2]);
//             }
//         }
//     }

//     // z-interp
//     for (int k = 0; k < len2; k++)
//     {
//         int k2 = (k + 1) / 2;
//         if (k % 2 == 0)
//         {
//             for (int j = 0; j < len2; j++)
//             {
//                 for (int i = 0; i < len2; i++)
//                 {
//                     datai[i + j * len2 + k * len2 * len2] = datay[i + j * len2 + k2 * len2 * len2];
//                 }
//             }
//         }
//         else
//         {
//             for (int j = 0; j < len2; j++)
//             {
//                 for (int i = 0; i < len2; i++)
//                 {
//                     datai[i + j * len2 + k * len2 * len2] = 0.5 * (datay[i + j * len2 + k2 * len2 * len2] + datay[i + j * len2 + (k2 - 1) * len2 * len2]);
//                 }
//             }
//         }
//     }
// }

// void dataCostCL(uint64_t *data, uint64_t *data2, float *results, int m, int n, int o, int len2, int step1, int hw, float quant, float alpha, int randnum)
// {
//     cout << "d" << flush;

//     int len = hw * 2 + 1;
//     len2 = pow(hw * 2 + 1, 3);

//     int sz = m * n * o;
//     int m1 = m / step1;
//     int n1 = n / step1;
//     int o1 = o / step1;
//     int sz1 = m1 * n1 * o1;

//     // cout<<"len2: "<<len2<<" sz1= "<<sz1<<"\n";

//     int quant2 = quant;

//     // const int hw2=hw*quant2; == pad1

//     int pad1 = quant2 * hw;
//     int pad2 = pad1 * 2;

//     int mp = m + pad2;
//     int np = n + pad2;
//     int op = o + pad2;
//     int szp = mp * np * op;
//     uint64_t *data2p = new uint64_t[szp];

//     for (int k = 0; k < op; k++)
//     {
//         for (int j = 0; j < np; j++)
//         {
//             for (int i = 0; i < mp; i++)
//             {
//                 data2p[i + j * mp + k * mp * np] = data2[max(min(i - pad1, m - 1), 0) + max(min(j - pad1, n - 1), 0) * m + max(min(k - pad1, o - 1), 0) * m * n];
//             }
//         }
//     }

//     int skipz = 1;
//     int skipx = 1;
//     int skipy = 1;
//     if (step1 > 4)
//     {
//         if (randnum > 0)
//         {
//             skipz = 2;
//             skipx = 2;
//         }
//         if (randnum > 1)
//         {
//             skipy = 2;
//         }
//     }
//     if (randnum > 1 & step1 > 7)
//     {
//         skipz = 3;
//         skipx = 3;
//         skipy = 3;
//     }
//     if (step1 == 4 & randnum > 1)
//         skipz = 2;

//     float maxsamp = ceil((float)step1 / (float)skipx) * ceil((float)step1 / (float)skipz) * ceil((float)step1 / (float)skipy);
//     // printf("randnum: %d, maxsamp: %d ",randnum,(int)maxsamp);

//     float alphai = (float)step1 / (alpha * (float)quant);

//     float alpha1 = 0.5 * alphai / (float)(maxsamp);

//     // uint64_t buffer[1000];

// #pragma omp parallel for
//     for (int z = 0; z < o1; z++)
//     {
//         for (int x = 0; x < n1; x++)
//         {
//             for (int y = 0; y < m1; y++)
//             {
//                 int z1 = z * step1;
//                 int x1 = x * step1;
//                 int y1 = y * step1;
//                 /*for(int k=0;k<step1;k++){
//                     for(int j=0;j<step1;j++){
//                         for(int i=0;i<step1;i++){
//                             buffer[i+j*step1+k*step1*step1]=data[i+y1+(j+x1)*m+(k+z1)*m*n];
//                         }
//                     }
//                 }*/

//                 for (int l = 0; l < len2; l++)
//                 {
//                     int out1 = 0;
//                     int zs = l / (len * len);
//                     int xs = (l - zs * len * len) / len;
//                     int ys = l - zs * len * len - xs * len;
//                     zs *= quant;
//                     xs *= quant;
//                     ys *= quant;
//                     int x2 = xs + x1;
//                     int z2 = zs + z1;
//                     int y2 = ys + y1;
//                     for (int k = 0; k < step1; k += skipz)
//                     {
//                         for (int j = 0; j < step1; j += skipx)
//                         {
//                             for (int i = 0; i < step1; i += skipy)
//                             {
//                                 // unsigned int t=buffer[i+j*STEP+k*STEP*STEP]^buf2p[i+j*mp+k*mp*np];
//                                 // out1+=(wordbits[t&0xFFFF]+wordbits[t>>16]);
//                                 uint64_t t1 = data[i + y1 + (j + x1) * m + (k + z1) * m * n]; // buffer[i+j*step1+k*step1*step1];
//                                 uint64_t t2 = data2p[i + j * mp + k * mp * np + (y2 + x2 * mp + z2 * mp * np)];
//                                 out1 += bitset<64>(t1 ^ t2).count();
//                             }
//                         }
//                     }
//                     results[(y + x * m1 + z * m1 * n1) * len2 + l] = out1 * alpha1;
//                 }
//             }
//         }
//     }

//     delete data2p;

//     return;
// }

// void warpImageCL(float *warped, float *im1, float *im1b, float *u1, float *v1, float *w1)
// {
//     int m = image_m;
//     int n = image_n;
//     int o = image_o;
//     int sz = m * n * o;

//     float ssd = 0;
//     float ssd0 = 0;
//     float ssd2 = 0;

//     interp3(warped, im1, u1, v1, w1, m, n, o, m, n, o, true);

//     for (int i = 0; i < m; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int k = 0; k < o; k++)
//             {
//                 ssd += pow(im1b[i + j * m + k * m * n] - warped[i + j * m + k * m * n], 2);
//                 ssd0 += pow(im1b[i + j * m + k * m * n] - im1[i + j * m + k * m * n], 2);
//             }
//         }
//     }

//     ssd /= m * n * o;
//     ssd0 /= m * n * o;
//     SSD0 = ssd0;
//     SSD1 = ssd;
// }

// void warpAffineS(short *warped, short *input, float *X, float *u1, float *v1, float *w1)
// {
//     int m = image_m;
//     int n = image_n;
//     int o = image_o;
//     int sz = m * n * o;
//     for (int k = 0; k < o; k++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 0; i < m; i++)
//             {

//                 float y1 = (float)i * X[0] + (float)j * X[1] + (float)k * X[2] + (float)X[3] + v1[i + j * m + k * m * n];
//                 float x1 = (float)i * X[4] + (float)j * X[5] + (float)k * X[6] + (float)X[7] + u1[i + j * m + k * m * n];
//                 float z1 = (float)i * X[8] + (float)j * X[9] + (float)k * X[10] + (float)X[11] + w1[i + j * m + k * m * n];
//                 int x = round(x1);
//                 int y = round(y1);
//                 int z = round(z1);

//                 // if(y>=0&x>=0&z>=0&y<m&x<n&z<o){
//                 warped[i + j * m + k * m * n] = input[min(max(y, 0), m - 1) + min(max(x, 0), n - 1) * m + min(max(z, 0), o - 1) * m * n];
//                 //}
//                 // else{
//                 //    warped[i+j*m+k*m*n]=0;
//                 //}
//             }
//         }
//     }
// }
// void warpAffine(float *warped, float *input, float *im1b, float *X, float *u1, float *v1, float *w1)
// {
//     int m = image_m;
//     int n = image_n;
//     int o = image_o;
//     int sz = m * n * o;

//     float ssd = 0;
//     float ssd0 = 0;
//     float ssd2 = 0;

//     for (int k = 0; k < o; k++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int i = 0; i < m; i++)
//             {

//                 float y1 = (float)i * X[0] + (float)j * X[1] + (float)k * X[2] + (float)X[3] + v1[i + j * m + k * m * n];
//                 float x1 = (float)i * X[4] + (float)j * X[5] + (float)k * X[6] + (float)X[7] + u1[i + j * m + k * m * n];
//                 float z1 = (float)i * X[8] + (float)j * X[9] + (float)k * X[10] + (float)X[11] + w1[i + j * m + k * m * n];
//                 int x = floor(x1);
//                 int y = floor(y1);
//                 int z = floor(z1);
//                 float dx = x1 - x;
//                 float dy = y1 - y;
//                 float dz = z1 - z;

//                 warped[i + j * m + k * m * n] = (1.0 - dx) * (1.0 - dy) * (1.0 - dz) * input[min(max(y, 0), m - 1) + min(max(x, 0), n - 1) * m + min(max(z, 0), o - 1) * m * n] +
//                                                 (1.0 - dx) * dy * (1.0 - dz) * input[min(max(y + 1, 0), m - 1) + min(max(x, 0), n - 1) * m + min(max(z, 0), o - 1) * m * n] +
//                                                 dx * (1.0 - dy) * (1.0 - dz) * input[min(max(y, 0), m - 1) + min(max(x + 1, 0), n - 1) * m + min(max(z, 0), o - 1) * m * n] +
//                                                 (1.0 - dx) * (1.0 - dy) * dz * input[min(max(y, 0), m - 1) + min(max(x, 0), n - 1) * m + min(max(z + 1, 0), o - 1) * m * n] +
//                                                 dx * dy * (1.0 - dz) * input[min(max(y + 1, 0), m - 1) + min(max(x + 1, 0), n - 1) * m + min(max(z, 0), o - 1) * m * n] +
//                                                 (1.0 - dx) * dy * dz * input[min(max(y + 1, 0), m - 1) + min(max(x, 0), n - 1) * m + min(max(z + 1, 0), o - 1) * m * n] +
//                                                 dx * (1.0 - dy) * dz * input[min(max(y, 0), m - 1) + min(max(x + 1, 0), n - 1) * m + min(max(z + 1, 0), o - 1) * m * n] +
//                                                 dx * dy * dz * input[min(max(y + 1, 0), m - 1) + min(max(x + 1, 0), n - 1) * m + min(max(z + 1, 0), o - 1) * m * n];
//             }
//         }
//     }

//     for (int i = 0; i < m; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int k = 0; k < o; k++)
//             {
//                 ssd += pow(im1b[i + j * m + k * m * n] - warped[i + j * m + k * m * n], 2);
//                 ssd0 += pow(im1b[i + j * m + k * m * n] - input[i + j * m + k * m * n], 2);
//             }
//         }
//     }

//     ssd /= m * n * o;
//     ssd0 /= m * n * o;
//     SSD0 = ssd0;
//     SSD1 = ssd;
// }


// void deeds(float *im1, float *im1b, float *warped1, float *flow, int m, int n, int o, float alpha, int levels, bool verbose)
// {
//     vector<int> grid_spacing = {8, 7, 6, 5, 4};
//     vector<int> search_radius = {8, 7, 6, 5, 4};
//     vector<int> quantisation = {5, 4, 3, 2, 1};

//     if (!verbose)
//         cout.rdbuf(nullptr);

//     cout << "Starting registration\n";
//     cout << "=============================================================\n";

//     RAND_SAMPLES = 1; // fixed/efficient random sampling strategy

//     image_m = m;
//     image_n = n;
//     image_o = o;

//     int sz = m * n * o;

//     // assume we are working with CT scans (add 1024 HU)
//     float thresholdF = -1024;
//     float thresholdM = -1024;

//     for (int i = 0; i < sz; i++)
//     {
//         im1b[i] -= thresholdF;
//         im1[i] -= thresholdM;
//     }

//     // READ AFFINE MATRIX from linearBCV if provided (else start from identity)

//     float *X = new float[16];

//     cout << "Starting with identity transform.\n";
//     fill(X, X + 16, 0.0f);
//     X[0] = 1.0f;
//     X[1 + 4] = 1.0f;
//     X[2 + 8] = 1.0f;
//     X[3 + 12] = 1.0f;

//     // PATCH-RADIUS FOR MIND/SSC DESCRIPTORS

//     vector<int> mind_step;
//     for (int i = 0; i < quantisation.size(); i++)
//     {
//         mind_step.push_back(floor(0.5f * (float)quantisation[i] + 1.0f));
//     }

//     cout << "MIND STEPS: " << mind_step[0] << " " << mind_step[1] << " " << mind_step[2] << " " << mind_step[3] << " " << mind_step[4] << endl;

//     int step1 = 0, hw1 = 0;
//     float quant1 = 0.0;

//     // set initial flow-fields to 0; i indicates backward (inverse) transform
//     // u is in x-direction (2nd dimension), v in y-direction (1st dim) and w in z-direction (3rd dim)
//     float *ux = new float[sz];
//     float *vx = new float[sz];
//     float *wx = new float[sz];
//     for (int i = 0; i < sz; i++)
//     {
//         ux[i] = 0.0;
//         vx[i] = 0.0;
//         wx[i] = 0.0;
//     }
//     int m2 = 0, n2 = 0, o2 = 0, sz2 = 0;
//     int m1 = 0, n1 = 0, o1 = 0, sz1 = 0;

//     m2 = m / grid_spacing[0];
//     n2 = n / grid_spacing[0];
//     o2 = o / grid_spacing[0];
//     sz2 = m2 * n2 * o2;
//     float *u1 = new float[sz2];
//     float *v1 = new float[sz2];
//     float *w1 = new float[sz2];
//     float *u1i = new float[sz2];
//     float *v1i = new float[sz2];
//     float *w1i = new float[sz2];
//     for (int i = 0; i < sz2; i++)
//     {
//         u1[i] = 0.0;
//         v1[i] = 0.0;
//         w1[i] = 0.0;
//         u1i[i] = 0.0;
//         v1i[i] = 0.0;
//         w1i[i] = 0.0;
//     }

//     float *warped0 = new float[m * n * o];
//     warpAffine(warped0, im1, im1b, X, ux, vx, wx);

//     uint64_t *im1_mind = new uint64_t[m * n * o];
//     uint64_t *im1b_mind = new uint64_t[m * n * o];
//     uint64_t *warped_mind = new uint64_t[m * n * o];

//     auto time1a = chrono::steady_clock::now();

//     //==========================================================================================

//     for (int level = 0; level < levels; level++)
//     {
//         quant1 = quantisation[level];

//         float prev = mind_step[max(level - 1, 0)]; // max(min(label_quant[max(level-1,0)],2.0f),1.0f);
//         float curr = mind_step[level];             // max(min(label_quant[level],2.0f),1.0f);

//         float timeMIND = 0;
//         float timeSmooth = 0;
//         float timeData = 0;
//         float timeTrans = 0;

//         if (level == 0 | prev != curr)
//         {
//             auto time1 = chrono::steady_clock::now();

//             descriptor(im1_mind, warped0, m, n, o, mind_step[level]); // im1 affine
//             descriptor(im1b_mind, im1b, m, n, o, mind_step[level]);

//             auto time2 = chrono::steady_clock::now();

//             timeMIND += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         }

//         step1 = grid_spacing[level];
//         hw1 = search_radius[level];

//         int len3 = pow(hw1 * 2 + 1, 3);
//         m1 = m / step1;
//         n1 = n / step1;
//         o1 = o / step1;
//         sz1 = m1 * n1 * o1;

//         float *costall = new float[sz1 * len3];
//         float *u0 = new float[sz1];
//         float *v0 = new float[sz1];
//         float *w0 = new float[sz1];
//         int *ordered = new int[sz1];
//         int *parents = new int[sz1];
//         float *edgemst = new float[sz1];

//         cout << "==========================================================\n";
//         cout << "Level " << level << " grid=" << step1 << " with sizes: " << m1 << "x" << n1 << "x" << o1 << " hw=" << hw1 << " quant=" << quant1 << "\n";
//         cout << "==========================================================\n";

//         // FULL-REGISTRATION FORWARDS
//         auto time1 = chrono::steady_clock::now();

//         upsampleDeformationsCL(u0, v0, w0, u1, v1, w1, m1, n1, o1, m2, n2, o2);
//         upsampleDeformationsCL(ux, vx, wx, u0, v0, w0, m, n, o, m1, n1, o1);
//         // float dist=landmarkDistance(ux,vx,wx,m,n,o,distsmm,casenum);
//         warpAffine(warped1, im1, im1b, X, ux, vx, wx);
//         u1 = new float[sz1];
//         v1 = new float[sz1];
//         w1 = new float[sz1];

//         auto time2 = chrono::steady_clock::now();

//         timeTrans += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "T" << flush;

//         time1 = chrono::steady_clock::now();

//         descriptor(warped_mind, warped1, m, n, o, mind_step[level]);

//         time2 = chrono::steady_clock::now();

//         timeMIND += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "M" << flush;

//         time1 = chrono::steady_clock::now();
//         dataCostCL(im1b_mind, warped_mind, costall, m, n, o, len3, step1, hw1, quant1, alpha, RAND_SAMPLES);
//         time2 = chrono::steady_clock::now();

//         timeData += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "D" << flush;
//         time1 = chrono::steady_clock::now();
//         primsGraph(im1b, ordered, parents, edgemst, step1, m, n, o);
//         regularisationCL(costall, u0, v0, w0, u1, v1, w1, hw1, step1, quant1, ordered, parents, edgemst);
//         time2 = chrono::steady_clock::now();
//         timeSmooth += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "S" << flush;

//         // FULL-REGISTRATION BACKWARDS
//         time1 = chrono::steady_clock::now();
//         upsampleDeformationsCL(u0, v0, w0, u1i, v1i, w1i, m1, n1, o1, m2, n2, o2);
//         upsampleDeformationsCL(ux, vx, wx, u0, v0, w0, m, n, o, m1, n1, o1);
//         warpImageCL(warped1, im1b, warped0, ux, vx, wx);
//         u1i = new float[sz1];
//         v1i = new float[sz1];
//         w1i = new float[sz1];
//         time2 = chrono::steady_clock::now();
//         timeTrans += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "T" << flush;
//         time1 = chrono::steady_clock::now();
//         descriptor(warped_mind, warped1, m, n, o, mind_step[level]);

//         time2 = chrono::steady_clock::now();
//         timeMIND += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "M" << flush;
//         time1 = chrono::steady_clock::now();
//         dataCostCL(im1_mind, warped_mind, costall, m, n, o, len3, step1, hw1, quant1, alpha, RAND_SAMPLES);
//         time2 = chrono::steady_clock::now();
//         timeData += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "D" << flush;
//         time1 = chrono::steady_clock::now();
//         primsGraph(warped0, ordered, parents, edgemst, step1, m, n, o);
//         regularisationCL(costall, u0, v0, w0, u1i, v1i, w1i, hw1, step1, quant1, ordered, parents, edgemst);
//         time2 = chrono::steady_clock::now();
//         timeSmooth += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
//         cout << "S" << flush;

//         cout << "\nTime: MIND=" << timeMIND << ", data=" << timeData << ", MST-reg=" << timeSmooth << ", transf.=" << timeTrans << "\n speed=" << 2.0 * (float)sz1 * (float)len3 / (timeData + timeSmooth) << " dof/s\n";

//         time1 = chrono::steady_clock::now();
//         consistentMappingCL(u1, v1, w1, u1i, v1i, w1i, m1, n1, o1, step1);
//         time2 = chrono::steady_clock::now();
//         float timeMapping = chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();

//         // cout<<"Time consistentMapping: "<<timeMapping<<"  \n";

//         // upsample deformations from grid-resolution to high-resolution (trilinear=1st-order spline)
//         float jac = jacobian(u1, v1, w1, m1, n1, o1, step1);

//         cout << "SSD before registration: " << SSD0 << " and after " << SSD1 << "\n";
//         m2 = m1;
//         n2 = n1;
//         o2 = o1;
//         cout << "\n";

//         delete u0;
//         delete v0;
//         delete w0;
//         delete costall;

//         delete parents;
//         delete ordered;
//     }
//     delete im1_mind;
//     delete im1b_mind;
//     //==========================================================================================
//     //==========================================================================================

//     auto time2a = chrono::steady_clock::now();

//     float timeALL = chrono::duration_cast<chrono::duration<float>>(time2a - time1a).count();

//     upsampleDeformationsCL(ux, vx, wx, u1, v1, w1, m, n, o, m1, n1, o1);

//     //float *flow = new float[sz1 * 3]; //Allocated outside
//     for (int i = 0; i < sz1; i++)
//     {
//         flow[i] = u1[i];
//         flow[i + sz1] = v1[i];
//         flow[i + sz1 * 2] = w1[i];
//         // flow[i+sz1*3]=u1i[i]; flow[i+sz1*4]=v1i[i]; flow[i+sz1*5]=w1i[i];
//     }

//     // WRITE OUTPUT DISPLACEMENT FIELD AND IMAGE
//     warpAffine(warped1, im1, im1b, X, ux, vx, wx);

//     for (int i = 0; i < sz; i++)
//     {
//         warped1[i] += thresholdM;
//     }

//     cout << "SSD before registration: " << SSD0 << " and after " << SSD1 << "\n";

//     cout << "Finished. Total time: " << timeALL << " sec." << endl;
// }
