#ifndef __ALPHABETATRACKER_H
#define __ALPHABETATRACKER_H

class AlphaBetaTracker
{
public:
    AlphaBetaTracker();
    void SetAlphaBeta(double a, double b);
    void SetState(double x, double y, double z, double yaw, double pitch, double roll);
    void GetCorrectedState(double *x, double *y, double *z, double *yaw, double *pitch, double *roll);
    void GetPredictedState(double *x, double *y, double *z, double *yaw, double *pitch, double *roll);
    void GetVelocity(double *dx, double *dy, double *dz, double *dyaw, double *dpitch, double *droll);
    bool Ready();
    void Reset();

private:
    double alpha, beta;
    double est_x, est_y, est_z, est_yaw, est_pitch, est_roll;
    double est_dx, est_dy, est_dz, est_dyaw, est_dpitch, est_droll;
    int count;
};

#endif
