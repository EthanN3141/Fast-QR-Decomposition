import matplotlib.pyplot as plt

# # RAW DATA. FROM strong_scaling_expts Generated by n=500,k=50
# Results for HQR (effect of p on time): [24.6985,15.1156,12.0377,13.0489,12.9767,12.56,12.983,11.4266,10.7466,10.2266,] 
# Results for Projection HQR (effect of p on time): [3.31029,2.08179,1.75147,1.88951,1.8516,1.76137,1.76995,1.50473,1.53587,1.50663,]
# Results for GS (effect of p on time): [3.52675,1.87203,1.39315,1.40803,1.37404,1.25192,1.21269,0.966653,0.893795,0.838286,]
# Results for projections GS (effect of p on time): [0.210413,0.139702,0.12016,0.115104,0.0992432,0.0799831,0.104362,0.095682,0.0861073,0.0755335]

# # RAW DATA. FROM weak_scaling_expts start n=300,k=200
# Results for HQR (weak scaling): [3.89604,5.60836,9.15138,17.3642,] 
# Results for Projection HQR (weak scaling): [4.63419,6.00896,9.28481,18.4299,]
# Results for GS (weak scaling): [0.621297,0.683745,0.928533,1.50592,]
# Results for projections GS (weak scaling): [0.943773,0.988357,1.44186,2.41289,] 

# # RAW DATA. FROM k_scaling_expts start n = 600, k = 10
# Results for Projection HQR (k scaling): [0.436564,1.73948,3.42857,4.76354,6.64558,8.04521,10.2196,12.6447,14.2347,15.7384,19.1627,22.0487,24.5086,27.1047,]
# Results for projections GS (k): [0.0596151,0.233322,0.569624,0.809074,0.987455,1.15082,1.5386,1.9557,2.00242,2.55174,3.09386,3.39156,3.94345,4.59902,]

def plot_strong_scaling_expts():
    HQR_p = [24.6985,15.1156,12.0377,13.0489,12.9767,12.56,12.983,11.4266,10.7466,10.2266]
    PHQR_p = [3.31029,2.08179,1.75147,1.88951,1.8516,1.76137,1.76995,1.50473,1.53587,1.50663]
    GS_p = [3.52675,1.87203,1.39315,1.40803,1.37404,1.25192,1.21269,0.966653,0.893795,0.838286]
    PGS_p = [0.210413,0.139702,0.12016,0.115104,0.0992432,0.0799831,0.104362,0.095682,0.0861073,0.0755335]

    x_axis_p = [1,2,3,4,5,6,7,8,9,10]

    plt.plot(x_axis_p, HQR_p)
    plt.title("Householder QR: processors vs time")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()

    plt.plot(x_axis_p, PHQR_p)
    plt.title("Householder QR with Projections: processors vs time")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()


    plt.plot(x_axis_p, GS_p)
    plt.title("MGS: processors vs time")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()

    plt.plot(x_axis_p, PGS_p)
    plt.title("MGS with Projections: processors vs time")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()

def plot_weak_scaling_expts():
    HQR_w = [3.89604,5.60836,9.15138,17.3642]
    PHQR_w = [4.63419,6.00896,9.28481,18.4299]
    GS_w = [0.621297,0.683745,0.928533,1.50592]
    PGS_w = [0.943773,0.988357,1.44186,2.41289] 

    x_axis_w = [1,2,4,8]

    plt.plot(x_axis_w, HQR_w)
    plt.title("Householder QR: constant p/size")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()

    plt.plot(x_axis_w, PHQR_w)
    plt.title("Householder QR with Projections: constant p/size")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()


    plt.plot(x_axis_w, GS_w)
    plt.title("MGS: constant p/size")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()

    plt.plot(x_axis_w, PGS_w)
    plt.title("MGS with Projections: constant p/size")
    plt.xlabel("number of processors")
    plt.ylabel("time (s)")
    plt.show()

def plot_k_scaling_expts():
    PHQR_p = [0.436564,1.73948,3.42857,4.76354,6.64558,8.04521,10.2196,12.6447,14.2347,15.7384,19.1627,22.0487,24.5086,27.1047]
    PGS_p = [0.0596151,0.233322,0.569624,0.809074,0.987455,1.15082,1.5386,1.9557,2.00242,2.55174,3.09386,3.39156,3.94345,4.59902]

    x_axis_p = [10,40,70,100,130,160,190,220,250,280,310,340,370,400]

    plt.plot(x_axis_p, PHQR_p)
    plt.title("Householder QR with Projections: rank vs time")
    plt.xlabel("rank")
    plt.ylabel("time (s)")
    plt.show()


    plt.plot(x_axis_p, PGS_p)
    plt.title("MGS with Projections: rank vs time")
    plt.xlabel("rank")
    plt.ylabel("time (s)")
    plt.show()


# plot_strong_scaling_expts()
# plot_weak_scaling_expts()
plot_k_scaling_expts()
