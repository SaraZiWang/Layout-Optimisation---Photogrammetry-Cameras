import RGBDCamClass as CamSim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import open3d as o3d
# import open3d_tutorial as o3dtut
from scipy.spatial.transform import Rotation as R
from scipy.optimize import dual_annealing
from numpy.linalg import inv
import socket
import os
import pandas as pd
import collections
import sys
import copy
import time
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import ctypes  # An included library with Python install.

def objFullEvaluation_num(CamL, CamR,AGV_x, AGV_y):

    evCamReward =  np.empty([len(AGV_x), len(AGV_y)])

    for ll in range(len(CamL)):
        command_x = 'Relocate.CamL(' + str(CamL[ll]) + ')'
        recv = PySend(command_x)
        for rr in range(len(CamR)):
            command_y = 'Relocate.CamR(' + str(CamR[rr]) + ')'
            recv = PySend(command_y)
            for nn in range(len(AGV_x)):
                x = AGV_x[nn]
                command_x = 'Relocate.AGV(1,' + str(x) + ')'
                recv = PySend(command_x)
                for kk in range(len(AGV_y)):
                    y = AGV_y[kk]
                    command_y = 'Relocate.AGV(0,' + str(y) + ')'
                    recv = PySend(command_y)


                    recv = PySend('TakeSnapshot')
                    if 'Snapshot saved at' in recv:
                        [fitness, coverage, unc] = CamSim.frameVisibility(False)

                        pv = 100
                        pq = 0.01
                        camReward = pv * fitness - pq * unc

                        unc_threshold = 500
                        if unc > unc_threshold:
                            camReward = 0

                        recv = PySend('Check reachability')
                        if recv == 'False':
                            reachability = 0
                            camReward = 0.
                        elif recv == 'True':
                            reachability = 1

                        evCamReward[nn, kk] = camReward

                        print('AGV at [' + str(x) + ', ' + str(y) + ']' + ', CamL at [' + str(
                            CamL[ll]) + '], CamR at [' + str(CamR[rr]) + '], Reward: ' + str(camReward)[:5])

            evCamReward.tofile('CamReward Results Grid/CamReward_OOI_' + str(ll) + str(rr) + '.dat')
def numSolution():
    nL = 5
    CamL = np.linspace(1280, 1300, num=nL, dtype=float)

    nR = 5
    CamR = np.linspace(1200, 1220, num=nR, dtype=float)

    nx = 6
    AGV_x = np.linspace(5106, 4356, num=nx, dtype=float)

    ny = 7
    AGV_y = np.linspace(25541, 24641, num=ny, dtype=float)

    # objFullEvaluation_num(CamL, CamR, AGV_x, AGV_y)

    AGV_x, AGV_y = np.meshgrid(AGV_x, AGV_y)

    CamReward = np.empty([ny, nx])

    fig, axs = plt.subplots(nL, nR)

    for ii in range(nL):
        for jj in range(nR):
            ax = axs[ii, jj]
            file = 'CamReward Results Grid/CamReward_OOI_' + str(ii) + str(jj) + '.dat'
            a = np.fromfile(file, dtype=float)
            for col in range(nx):
                CamReward[:, col] = a[col * ny:(col + 1) * ny]

            ax.axis('off')
            c = ax.pcolormesh(AGV_x, AGV_y, CamReward, cmap='RdBu', vmin=16, vmax=37)
            ax.invert_xaxis()
            # ax.set_title('L:'+str(CamL[ii])[:4]+' R:'+str(CamR[jj])[:4]+' Max:'+str(a.max())[:4], fontsize=10)
            ax.set_title('Max:' + str(CamReward.max())[:5], fontsize=12)

    cbar = fig.colorbar(c, ax=axs[:, -1])
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(top=0.88,
    bottom=0.11,
    left=0.467,
    right=0.871,
    hspace=0.28,
    wspace=0.2)
    plt.show()
def animationPlot(data,ppf):
    x, y = data[:, 0], data[:, 1]
    fig, ax = plt.subplots()
    ax.set_xlim(3900, 4510)
    ax.set_ylim(24640, 26145)
    graph, = plt.plot([], [], 'ro-',linewidth=0.5, markersize=5)
    def init():
        return graph,
    def animate(i):
        graph.set_data(x[:i + ppf - 1], y[:i + ppf - 1])
        return graph,

    ani = animation.FuncAnimation(fig, animate, frames=range(np.rint(len(x)/ppf).astype(int)), interval=100,
                        init_func=init, blit=True)
    plt.show()
def PySend(sendmsg):
    global conn
    # send data to the client
    conn.send(sendmsg.encode())
    # receive data stream
    recvdata = conn.recv(1024).decode()
    # print("from PS user: " + str(recvdata))
    return recvdata

def objective(v):
    global f
    global conn

    AGV_x, AGV_y, CamL, CamR = v

    command_x = 'Relocate.AGV(1,' + str(AGV_x) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.AGV(0,' + str(AGV_y) + ')'
    recv = PySend(command_y)

    command_x = 'Relocate.CamL('+str(CamL) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.CamR(' + str(CamR) + ')'
    recv = PySend(command_y)

    recv = PySend('TakeSnapshot')

    print('AGV at [' + str(AGV_x)[:8] + ', ' + str(AGV_y)[:9] + ']'+', CamL at ['+str(CamL)[:8]+'], CamR at ['+str(CamR)[:8]+']')

    if 'Snapshot saved at' in recv:
        # print("Output files evaluation in progress...")
        [fitness, coverage, unc] = CamSim.frameVisibility()
        # print(
        #     "CAD fitness ratio: " + str(fitness * 100)[0:5] + "%, Voxel coverage ratio: " + str(coverage * 100)[
        #                                                                                  0:5] + "%.")
        # print('uncertainty: ' + str(unc))

        pv = 100
        pq = 0.01
        camReward = pv*fitness - pq*unc

        unc_threshold = 500
        if unc > unc_threshold:
            camReward = 0

    recv = PySend('Check reachability')
    if recv == 'False':
        reachability = 0
        camReward = 0.
    elif recv == 'True':
        reachability = 1

    f.write(str(AGV_x) + ' ' + str(AGV_y) + ' ' + str(CamL) + ' ' + str(CamR) + ' ' + str(fitness) + ' ' + str(reachability) + ' ' + str(unc) + ' '+ str(camReward)+ '\n')

    return -camReward
def objective_BO(AGV_x, AGV_y, CamL, CamR):
    global f
    global conn

    command_x = 'Relocate.AGV(1,' + str(AGV_x) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.AGV(0,' + str(AGV_y) + ')'
    recv = PySend(command_y)

    command_x = 'Relocate.CamL('+str(CamL) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.CamR(' + str(CamR) + ')'
    recv = PySend(command_y)

    recv = PySend('TakeSnapshot')

    # print('AGV at [' + str(AGV_x) + ', ' + str(AGV_y) + ']'+', CamL at ['+str(CamL)+'], CamR at ['+str(CamR)+']')

    if 'Snapshot saved at' in recv:
        # print("Output files evaluation in progress...")
        [fitness, coverage, unc] = CamSim.frameVisibility(False)
        # print(
        #     "CAD fitness ratio: " + str(fitness * 100)[0:5] + "%, Voxel coverage ratio: " + str(coverage * 100)[
        #                                                                                  0:5] + "%.")
        # print('uncertainty: ' + str(unc))

        pv = 100
        pq = 0.01
        camReward = pv*fitness - pq*unc

        unc_threshold = 500
        if unc > unc_threshold:
            camReward = 0

    recv = PySend('Check reachability')
    if recv == 'False':
        reachability = 0
        # camReward = 0.
    elif recv == 'True':
        reachability = 1

    f.write(str(AGV_x) + ' ' + str(AGV_y) + ' ' + str(CamL) + ' ' + str(CamR) + ' ' + str(fitness) + ' ' + str(reachability) + ' ' + str(unc) + ' '+ str(camReward)+ '\n')

    return camReward

def objective_cov(v):
    global f
    global conn

    AGV_x, AGV_y, CamL, CamR = v

    command_x = 'Relocate.AGV(1,' + str(AGV_x) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.AGV(0,' + str(AGV_y) + ')'
    recv = PySend(command_y)

    command_x = 'Relocate.CamL('+str(CamL) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.CamR(' + str(CamR) + ')'
    recv = PySend(command_y)

    recv = PySend('TakeSnapshot')

    print('AGV at [' + str(AGV_x)[:8] + ', ' + str(AGV_y)[:9] + ']'+', CamL at ['+str(CamL)[:8]+'], CamR at ['+str(CamR)[:8]+']')

    if 'Snapshot saved at' in recv:
        # print("Output files evaluation in progress...")
        [_, coverage, unc] = CamSim.frameVisibility(True)
        # print(
        #     "CAD fitness ratio: " + str(fitness * 100)[0:5] + "%, Voxel coverage ratio: " + str(coverage * 100)[
        #                                                                                  0:5] + "%.")
        # print('uncertainty: ' + str(unc))

        pv = 100
        pq = 0.01
        camReward = pv*coverage - pq*unc

        unc_threshold = 500
        if unc > unc_threshold:
            camReward = 0

    recv = PySend('Check reachability')
    if recv == 'False':
        reachability = 0
        camReward = 0.
    elif recv == 'True':
        reachability = 1

    f.write(str(AGV_x) + ' ' + str(AGV_y) + ' ' + str(CamL) + ' ' + str(CamR) + ' ' + str(coverage) + ' ' + str(reachability) + ' ' + str(unc) + ' '+ str(camReward)+ '\n')

    return -camReward
def objective_BO_cov(AGV_x, AGV_y, CamL, CamR):
    global f
    global conn

    command_x = 'Relocate.AGV(1,' + str(AGV_x) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.AGV(0,' + str(AGV_y) + ')'
    recv = PySend(command_y)

    command_x = 'Relocate.CamL('+str(CamL) + ')'
    recv = PySend(command_x)

    command_y = 'Relocate.CamR(' + str(CamR) + ')'
    recv = PySend(command_y)

    recv = PySend('TakeSnapshot')

    # print('AGV at [' + str(AGV_x) + ', ' + str(AGV_y) + ']'+', CamL at ['+str(CamL)+'], CamR at ['+str(CamR)+']')

    if 'Snapshot saved at' in recv:
        # print("Output files evaluation in progress...")
        [_, coverage, unc] = CamSim.frameVisibility(True)
        # print(
        #     "CAD fitness ratio: " + str(fitness * 100)[0:5] + "%, Voxel coverage ratio: " + str(coverage * 100)[
        #                                                                                  0:5] + "%.")
        # print('uncertainty: ' + str(unc))

        pv = 100
        pq = 0.01
        camReward = pv*coverage - pq*unc

        unc_threshold = 500
        if unc > unc_threshold:
            camReward = 0

    recv = PySend('Check reachability')
    if recv == 'False':
        reachability = 0
        camReward = 0.
    elif recv == 'True':
        reachability = 1

    f.write(str(AGV_x) + ' ' + str(AGV_y) + ' ' + str(CamL) + ' ' + str(CamR) + ' ' + str(coverage) + ' ' + str(reachability) + ' ' + str(unc) + ' '+ str(camReward)+ '\n')

    return camReward

def solution_SA():
    # define the bounds on the search
    bounds = [[x_min, x_max], [y_min, y_max], [CamL_min, CamL_max], [CamR_min, CamR_max]]

    def printx(x, e, context):
        print('new best at :['+str(x[0])[:7]+', '+str(x[1])[:8]+', '+str(x[2])[:7]+', '+str(x[3])[:7]+']' + ', fitness:' + str(e)[:6])
        f2.write(str(x[0])+' '+str(x[1])+' '+str(x[2])+' '+str(x[3])+' '+str(e)+'\n')

    progress = 1
    for ii in range(10):
    # perform the dual annealing search
        print('-----------------------------------' + str(progress)+'-----------------------------------')
        result = dual_annealing(objective, bounds, maxfun=100, accept=QA, visit=2.62, no_local_search=True,callback=printx)
        progress +=1

    # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # # evaluate solution
    # solution = result['x']
    # evaluation = objective(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    #
    # recv = PySend('Done')  # Ending comms
    # conn.close()  # close the connection
    # print('Computational Time: ' + str((time.time() - t0) / 60)[:5])
def solution_GA():
    # decode bitstring to numbers
    def decode(bounds, n_bits, bitstring):
        decoded = list()
        largest = 2 ** n_bits
        for i in range(len(bounds)):
            # extract the substring
            start, end = i * n_bits, (i * n_bits) + n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
            # store
            decoded.append(value)
        return decoded

    # tournament selection
    def selection(pop, scores, k=3):
        # first random selection
        selection_ix = np.random.randint(len(pop))
        for ix in np.random.randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # crossover two parents to create two children
    # Variant0 - crossover the entire vector in its decoded form
    def crossover0(n_input, n_bits, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # Variant1 - crossover within each variable
    def crossover1(n_input, n_bits, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, n_bits - 1)
            # perform crossover
            # c1 = p1[:pt] + p2[pt:]
            # c2 = p2[:pt] + p1[pt:]
            for nn in range(n_input):
                c1[nn*n_bits: (nn+1)*n_bits] = p1[nn*n_bits: nn*n_bits+pt] + p2[nn*n_bits+pt: (nn+1)*n_bits]
                c2[nn*n_bits: (nn+1)*n_bits] = p2[nn*n_bits: nn*n_bits+pt] + p1[nn*n_bits+pt: (nn+1)*n_bits]
        return [c1, c2]

    # Variant2 - crossover each variable
    def crossover2(n_input, n_bits, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, n_input)
            # perform crossover
            c1 = p1[:pt*n_bits] + p2[pt*n_bits:]
            c2 = p2[:pt*n_bits] + p1[pt*n_bits:]
        return [c1, c2]

    # mutation operator
    def mutation(bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if np.random.rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

    # genetic algorithm
    def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
        # initial population of random bitstring
        pop = [np.random.randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
        # keep track of best solution
        best, best_eval = pop[0], objective(decode(bounds, n_bits, pop[0]))
        # enumerate generations
        for gen in range(n_iter):
            # decode population
            decoded = [decode(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective(d) for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %f" % (gen, decoded[i], scores[i]))
                best_decoded = decode(bounds, n_bits, best)
                f2.write(str(best_decoded[0]) + ' ' + str(best_decoded[1]) + ' ' + str(best_decoded[2]) + ' ' + str(best_decoded[3]) + ' ' + str(best_eval) + ' ' + '\n')

            # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in crossover2(len(bounds), n_bits, p1, p2, r_cross):
                    # mutation
                    mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
        return [best, best_eval]

    # define the bounds on the search
    bounds = [[x_min, x_max], [y_min, y_max], [CamL_min, CamL_max], [CamR_min, CamR_max]]

    # define the population size
    n_pop = Np
    # define the total iterations
    n_iter = int(100/Np)
    # bits per variable
    n_bits = 8
    # crossover rate
    r_cross = Rc
    # mutation rate
    r_mut = 1/(float(n_bits) * len(bounds))
    progress = 1
    for _ in range(10):
        # perform the genetic algorithm search
        print('-----------------------------------' + str(progress)+'-----------------------------------')
        best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
        progress += 1
def solution_BO():
    # Bounded region of parameter space
    bounds = {'AGV_x': (x_min, x_max), 'AGV_y': (y_min, y_max), 'CamL': (CamL_min, CamL_max), 'CamR': (CamR_min, CamR_max)}
    util = UtilityFunction(kind='ucb',
                           kappa=kappa,
                           xi=0.0,
                           kappa_decay=1,
                           kappa_decay_delay=0)
    progress = 1
    for _ in range(10):
        print('-----------------------------------' + str(progress) + '-----------------------------------')
        optimizer = BayesianOptimization(f=objective_BO, pbounds=bounds, random_state=None, verbose=2, allow_duplicate_points=True)
        optimizer.maximize(init_points=10, n_iter=90, acquisition_function=util)
        progress += 1

if __name__=='__main__':

    """Establish Connection"""
    # get the hostname
    host = socket.gethostname()
    port = 2023  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    server_socket.bind(('127.0.0.1', port))  # bind host address and port together
    server_socket.listen(2)

    # while True:
    print("Waiting for connection...")
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))

    recv = PySend('Connected?')

    # define range for input
    x_min, x_max = 4356., 5106.
    y_min, y_max = 24641., 25541.
    CamL_min, CamL_max = 1280., 1300.
    CamR_min, CamR_max = 1200., 1220.

    """Numerical Solution"""
    # numSolution()
    # numSolutionAGV()


    """Simulated Annealing Implementation"""

    # for QA in [-5,-10,-1000]:
    #     t0 = time.time()
    #     fname = "Results/SAResultsQa"+str(QA)+".txt"
    #     f2name = "Results/SABestResultsQa"+str(QA)+".txt"
    #     f = open(fname, "w")
    #     f2 = open(f2name,'w')
    #     solution_SA()
    #     f2.close()
    #     f.close()
    #     print('Computational Time: ' + str((time.time() - t0) / 60)[:5])

    """ Genetic Algorithm Implementation """
    #
    # for Rc in [0.01,0.5,0.9]:
    #     Np=10
    #     t0 = time.time()
    #     fname = "Results/GAResultsRc_"+str(Np)+str(Rc)[-1]+".txt"
    #     f2name = "Results/GAbestResultsRc_"+str(Np)+str(Rc)[-1]+".txt"
    #     f = open(fname, "w")
    #     f2 = open(f2name,'w')
    #     solution_GA()
    #     f2.close()
    #     f.close()
    #     print('Computational Time: ' + str((time.time() - t0) / 60)[:5])
    #
    # for Np in [4,20]:
    #     Rc=0.9
    #     t0 = time.time()
    #     fname = "Results/GAResultsRc_"+str(Np)+str(Rc)[-1]+".txt"
    #     f2name = "Results/GAbestResultsRc_"+str(Np)+str(Rc)[-1]+".txt"
    #     f = open(fname, "w")
    #     f2 = open(f2name,'w')
    #     solution_GA()
    #     f2.close()
    #     f.close()
    #     print('Computational Time: ' + str((time.time() - t0) / 60)[:5])

    """ Bayesian Optimisation Implementation """
    for kappa in [0.01,5,10]:
        t0 = time.time()
        fname = "Results/BOcovResults_NSKD_kappa"+str(kappa)[-1]+"_10_new.txt"
        f = open(fname, "w")
        solution_BO()
        f.close()
        print('Computational Time: ' + str((time.time() - t0) / 60)[:5])

    recv = PySend('Done')
    conn.close()  # close the connection

