#include <iostream>
#include <assert.h>
#include <cmath>

#include "FeedForwardNeuralNetwork.hpp"



int main(){
   using namespace std;

   FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
   ffnn->pushHiddenLayer(4);
   ffnn->connectFFNN();

   ffnn->addFirstDerivativeSubstrate();
   ffnn->addSecondDerivativeSubstrate();
   ffnn->addVariationalFirstDerivativeSubstrate();
   ffnn->addCrossFirstDerivativeSubstrate();


   double x[2] = {1.7, -0.2};
   double dx = 0.00001;
   double x1[2];


   ffnn->setInput(x);
   ffnn->FFPropagate();
   double fx = ffnn->getOutput(0);
   double fy = ffnn->getOutput(1);
   //cout << "fx = " << fx << endl;
   //cout << "fy = " << fy << endl;



   // --- first and second derivative in respect to first input

   ffnn->setInput(x);
   ffnn->FFPropagate();
   double anal_dfxdx = ffnn->getFirstDerivative(0, 0);
   double anal_dfydx = ffnn->getFirstDerivative(1, 0);
   double anal_d2fxdx2 = ffnn->getSecondDerivative(0, 0);
   double anal_d2fydx2 = ffnn->getSecondDerivative(1, 0);

   x1[0] = x[0] + dx;
   x1[1] = x[1];
   ffnn->setInput(x1);
   ffnn->FFPropagate();
   double fx1 = ffnn->getOutput(0);
   double fy1 = ffnn->getOutput(1);
   //cout << "fx1 = " << fx1 << endl;
   //cout << "fy1 = " << fy1 << endl;

   x1[0] = x[0] - dx;
   x1[1] = x[1];
   ffnn->setInput(x1);
   ffnn->FFPropagate();
   double fxm1 = ffnn->getOutput(0);
   double fym1 = ffnn->getOutput(1);
   //cout << "fxm1 = " << fxm1 << endl;
   //cout << "fym1 = " << fym1 << endl;
   //cout << endl;

   double num_dfxdx = (fx1-fx)/dx;
   double num_dfydx = (fy1-fy)/dx;

   //cout << "anal_dfxdx = " << anal_dfxdx << endl;
   //cout << "num_dfxdx = " << num_dfxdx << endl;
   //cout << endl;
   assert(abs(anal_dfxdx-num_dfxdx) < 0.001);

   //cout << "anal_dfydx = " << anal_dfydx << endl;
   //cout << "num_dfydx = " << num_dfydx << endl;
   //cout << endl;
   assert(abs(anal_dfydx-num_dfydx) < 0.001);

   double num_d2fxdx2 = (fx1-2.*fx+fxm1)/(dx*dx);
   double num_d2fydx2 = (fy1-2.*fy+fym1)/(dx*dx);

   //cout << "anal_d2fxdx2 = " << anal_d2fxdx2 << endl;
   //cout << "num_d2fxdx2 = " << num_d2fxdx2 << endl;
   //cout << endl;
   assert(abs(anal_d2fxdx2-num_d2fxdx2) < 0.001);

   //cout << "anal_d2fydx2 = " << anal_d2fydx2 << endl;
   //cout << "num_d2fydx2 = " << num_d2fydx2 << endl;
   //cout << endl;
   assert(abs(anal_d2fydx2-num_d2fydx2) < 0.001);



   // --- first and second derivative in respect to second input

   ffnn->setInput(x);
   ffnn->FFPropagate();
   anal_dfxdx = ffnn->getFirstDerivative(0, 1);
   anal_dfydx = ffnn->getFirstDerivative(1, 1);
   anal_d2fxdx2 = ffnn->getSecondDerivative(0, 1);
   anal_d2fydx2 = ffnn->getSecondDerivative(1, 1);

   x1[0] = x[0];
   x1[1] = x[1] + dx;
   ffnn->setInput(x1);
   ffnn->FFPropagate();
   fx1 = ffnn->getOutput(0);
   fy1 = ffnn->getOutput(1);
   //cout << "fx1 = " << fx1 << endl;
   //cout << "fy1 = " << fy1 << endl;

   x1[0] = x[0];
   x1[1] = x[1] - dx;
   ffnn->setInput(x1);
   ffnn->FFPropagate();
   fxm1 = ffnn->getOutput(0);
   fym1 = ffnn->getOutput(1);
   //cout << "fxm1 = " << fxm1 << endl;
   //cout << "fym1 = " << fym1 << endl;
   //cout << endl;

   num_dfxdx = (fx1-fx)/dx;
   num_dfydx = (fy1-fy)/dx;

   //cout << "anal_dfxdx = " << anal_dfxdx << endl;
   //cout << "num_dfxdx = " << num_dfxdx << endl;
   //cout << endl;
   //assert(abs(anal_dfxdx-num_dfxdx) < 0.001);

   //cout << "anal_dfydx = " << anal_dfydx << endl;
   //cout << "num_dfydx = " << num_dfydx << endl;
   //cout << endl;
   assert(abs(anal_dfydx-num_dfydx) < 0.001);

   num_d2fxdx2 = (fx1-2.*fx+fxm1)/(dx*dx);
   num_d2fydx2 = (fy1-2.*fy+fym1)/(dx*dx);

   //cout << "anal_d2fxdx2 = " << anal_d2fxdx2 << endl;
   //cout << "num_d2fxdx2 = " << num_d2fxdx2 << endl;
   //cout << endl;
   assert(abs(anal_d2fydx2-num_d2fydx2) < 0.001);

   //cout << "anal_d2fydx2 = " << anal_d2fydx2 << endl;
   //cout << "num_d2fydx2 = " << num_d2fydx2 << endl;
   //cout << endl;
   assert(abs(anal_d2fydx2-num_d2fydx2) < 0.001);



   // --- variational derivative

   double * anal_dfxdbeta = new double[ffnn->getNBeta()];
   double * anal_dfydbeta = new double[ffnn->getNBeta()];

   ffnn->setInput(x);
   ffnn->FFPropagate();
   for (int i=0; i<ffnn->getNBeta(); ++i){
      anal_dfxdbeta[i] = ffnn->getVariationalFirstDerivative(0, i);
      anal_dfydbeta[i] = ffnn->getVariationalFirstDerivative(1, i);
   }


   for (int i=0; i<ffnn->getNBeta(); ++i){
      const double orig_beta = ffnn->getBeta(i);
      ffnn->setBeta(i, orig_beta+dx);
      ffnn->FFPropagate();
      fx1 = ffnn->getOutput(0);
      fy1 = ffnn->getOutput(1);

      const double num_dfxdbeta = (fx1-fx)/dx;
      const double num_dfydbeta = (fy1-fy)/dx;

      //cout << "i_beta = " << i << endl;
      //cout << "anal_dfxdbeta = " << anal_dfxdbeta[i] << endl;
      //cout << "num_dfxdbeta = " << num_dfxdbeta << endl;
      //cout << endl;
      assert(abs(anal_dfxdbeta[i]-num_dfxdbeta) < 0.001);

      //cout << "anal_dfydbeta = " << anal_dfydbeta[i] << endl;
      //cout << "num_dfydbeta = " << num_dfydbeta << endl;
      //cout << endl;
      assert(abs(anal_dfydbeta[i]-num_dfydbeta) < 0.001);

      ffnn->setBeta(i, orig_beta);
   }



   // --- cross derivatives

   double ** anal_dfxdxdbeta = new double*[ffnn->getNInput()];
   for (int i=0; i<ffnn->getNInput(); ++i){
       anal_dfxdxdbeta[i] = new double[ffnn->getNBeta()];
   }
   double ** anal_dfydxdbeta = new double*[ffnn->getNInput()];
   for (int i=0; i<ffnn->getNInput(); ++i){
       anal_dfydxdbeta[i] = new double[ffnn->getNBeta()];
   }

   ffnn->setInput(x);
   ffnn->FFPropagate();
   ffnn->getCrossFirstDerivative(0, anal_dfxdxdbeta);
   ffnn->getCrossFirstDerivative(1, anal_dfydxdbeta);

   for (int i1d=0; i1d<ffnn->getNInput(); ++i1d){
       for (int iv1d=0; iv1d<ffnn->getNBeta(); ++iv1d){
           const double orig_x = x[i1d];
           const double orig_beta = ffnn->getBeta(iv1d);

           ffnn->setInput(i1d, orig_x);
           ffnn->setBeta(iv1d, orig_beta+dx);
           ffnn->FFPropagate();
           const double fxdbeta = ffnn->getOutput(0);
           const double fydbeta = ffnn->getOutput(1);

           ffnn->setInput(i1d, orig_x+dx);
           ffnn->setBeta(iv1d, orig_beta);
           ffnn->FFPropagate();
           const double fxdx = ffnn->getOutput(0);
           const double fydx = ffnn->getOutput(1);

           ffnn->setInput(i1d, orig_x+dx);
           ffnn->setBeta(iv1d, orig_beta+dx);
           ffnn->FFPropagate();
           const double fxdxdbeta = ffnn->getOutput(0);
           const double fydxdbeta = ffnn->getOutput(1);

           const double num_dfxdxdbeta = (fxdxdbeta - fxdx - fxdbeta + fx)/(dx*dx);
           const double num_dfydxdbeta = (fydxdbeta - fydx - fydbeta + fy)/(dx*dx);

           cout << "anal_dfxdxdbeta[" << i1d << "][" << iv1d << "]    " << anal_dfxdxdbeta[i1d][iv1d] << endl;
           cout << " --- > num_dfxdxdbeta    " << num_dfxdxdbeta << endl;
           cout << "anal_dfydxdbeta[" << i1d << "][" << iv1d << "]    " << anal_dfydxdbeta[i1d][iv1d] << endl;
           cout << " --- > num_dfydxdbeta    " << num_dfydxdbeta << endl;
           cout << endl;

           ffnn->setInput(i1d, orig_x);
           ffnn->setBeta(iv1d, orig_beta);
       }
   }


   delete ffnn;

   return 0;
}
