import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spe
from matplotlib.lines import Line2D
import scipy.integrate as integrate
import copy
import time
import itertools


class MatchingCases:
    def __init__(self, n_firms=3, n_workers=10):
        self.n_firms = n_firms
        self.m = n_workers

    def case_iterator_i(self, case, i_firm):
        """
        :list case: a screening case up to firm i-1, e.g. [[4], [1], [2,3]]
        :param i_firm: index of firm i for which to build cases
        :return cases_i: list of cases built upon case
        """
        m = self.m
        # append firm to all possible screens
        cases_i = []
        for w in range(len(case)):
            case_i = copy.deepcopy(case)
            case_i[w].append(i_firm)
            cases_i.append(case_i)

        # add case that firm did not screen worker s
        cases_i.append(copy.deepcopy(case))
        # add case that firm screens previously unscreened worker
        if len(case) < m:
            case.append([i_firm])
            cases_i.append(case)
        return cases_i

    def case_builder_i(self, cases_prev, i_firm):
        """
        :param i_firm: firm index for which to build all cases
        :param cases_prev: all previous cases
        :return all_cases_i: list of all cases for firm i
        """
        all_cases_i = []
        for c in cases_prev:
            new_cases = self.case_iterator_i(c, i_firm)
            all_cases_i.extend(new_cases)
        return all_cases_i

    def case_builder(self):
        """
        :return dict cases: dictionary with all cases where key is firm index
                        {1: [[[n_firms]], [[n_firms], [1]], [[n_firms, 1]]],
                         2: ...
                         3: ...}
        """
        # initial case where only firm n_firms is assigned to a worker
        all_cases = {1: [[[self.n_firms]], [[self.n_firms, 1]], [[self.n_firms], [1]]]}

        for i in range(2, self.n_firms):
            all_cases_i = self.case_builder_i(copy.deepcopy(all_cases[i-1]), i)
            all_cases[i] = all_cases_i
        return all_cases


class ExpValues:
    def __init__(self, n_grid=100, n_firms=3, n_workers=10):

        # number of gridpoints for integration
        self.n_grid = n_grid

        # number of firms
        self.n_firms = n_firms

        # number of workers
        self.n_workers = n_workers

        # indices for lower triangle (will only consider s1>=s2)
        self.ilt1, self.ilt2 = np.tril_indices(self.n_grid)

        # number of workers
        if n_workers > 2:
            self.m = n_workers
        else:
            print('Choose larger number of workers.')

        self.m2 = self.binom(self.m, 2)

        # initialize acceptance probabilities for each firm above i given s1,s2
        # n_firms x n_grid x n_grid
        self.firmOfferProbs_s1 = np.zeros((self.n_firms, self.n_grid, self.n_grid))
        self.firmOfferProbs_s2 = np.zeros((self.n_firms, self.n_grid, self.n_grid))

        # initialize offer decision matrix for each firm i given they observe s1,s2
        # 1: make offer to s1;
        # 0: make offer to s2
        self.firmOfferActions = np.zeros((self.n_firms, self.n_grid, self.n_grid))
        # firm 1 always makes offer to s1
        self.firmOfferActions[0, self.ilt1, self.ilt2] = 1

    def binom(self, m, n):
        """
        :return: binomial coefficient (m,n)
        """
        return spe.comb(m, n, exact=True)

# functions to build expected values ######################################################

    def prob_case(self, ix_s1, ix_s2, case, firm_i, s):
        """
        :param ix_s1: First screen of a firm worse than firm_i
        :param ix_s2: Second screen of a firm worse than firm_i
        :list case: case with n_firms as firm n_firms sees it for firm firm_i
        :param firm_i: Lowest firm that is seen as a competitor by the worse firm
        :param s: 's1' or 's2': which screen under consideration by firm n
        :return: Prob. that no firm until firm_i (included) made offer to s in case case times probability of the case
        """

        # count no. of firms below which screened s
        n_s_screened = -1  # do not include firm n
        for w in case:
            n_s_screened += len(w)

        prob = ((self.m-1)/self.m2)**n_s_screened * ((self.m-2)/self.m)**(firm_i + 1-n_s_screened)
        # This is the probability of n_s_screened concrete firms until firm_i having screened s

        # factor for firms who screen exactly s1, s2
        n_screen_s1_s2 = len(case[0])-1
        prob *= (1/(self.m-1))**n_screen_s1_s2 * ((self.m - 2)/(self.m - 1))**(n_s_screened - n_screen_s1_s2)
        for k in range(1, len(case[0])):
            prob *= ((s == 's1')*(1-self.firmOfferActions[case[0][k]-1, ix_s1, ix_s2])
                     + (s == 's2')*self.firmOfferActions[case[0][k]-1, ix_s1, ix_s2])

        # factor for firms who screen different but identical second
        summedlength = n_screen_s1_s2        # the cardinality of G^1, G^2, ..., G^(l-1), when G^l is considered
        for l in range(1, len(case)):
            summedlength += len(case[l])
            prob *= (1/(self.m - l - 1))**(len(case[l]) - 1) * \
                    ((self.m - l - 2)/(self.m - l - 1))**(n_s_screened - summedlength)
            joint_indic_lower = np.ones(self.n_grid)  # measures the amount of lower screens to which a firm would
            # not make an offer, for a given higher screen s
            joint_indic_higher = np.ones(self.n_grid)  # measures the amount of higher screens to which a firm would
            # not make an offer, for a given lower screen s
            ix_s = 0
            for i in case[l]:
                joint_indic_lower = np.ones(self.n_grid)
                joint_indic_higher = np.ones(self.n_grid)
                if s == 's1':
                    joint_indic_lower *= 1 - self.firmOfferActions[i-1, ix_s1, :]
                    joint_indic_higher *= self.firmOfferActions[i-1, :, ix_s1]
                    ix_s = ix_s1
                if s == 's2':
                    joint_indic_lower *= 1 - self.firmOfferActions[i-1, ix_s2, :]
                    joint_indic_higher *= self.firmOfferActions[i-1, :, ix_s2]
                    ix_s = ix_s2

            prob *= (np.trapz(joint_indic_lower[:ix_s], dx=(1/self.n_grid)) + np.trapz(joint_indic_higher[ix_s:],
                                                                                       dx=(1/self.n_grid)))
        return prob

    def no_probs_cases(self, ix_s1, ix_s2, all_cases, firm_i, s):
        """
        param ix_s1:    index of screen s1
        param ix_s2:    index of screen s2
        list all_cases: list with all screening possibilities
        param firm_i:   firm i
        param s:        's1' or 's2': which screen under consideration by firm n
        :return: Probability that all firms until firm_i did not make an offer to s
        """
        offer_prob = 0
        for c in all_cases[firm_i + 1]:
            offer_prob += self.prob_case(ix_s1, ix_s2, c, firm_i, s)
        return offer_prob

    def prob_no_offer(self, all_cases, firm_i):
        """
        return
        Calculate prob matrices that firm up to i makes no offer to s1, s2
        """
        v_no_probs_cases = np.vectorize(self.no_probs_cases)

        self.firmOfferProbs_s1[firm_i, self.ilt1, self.ilt2] = v_no_probs_cases(self.ilt1, self.ilt2,
                                                                                all_cases, firm_i, 's1')
        self.firmOfferProbs_s2[firm_i, self.ilt1, self.ilt2] = v_no_probs_cases(self.ilt1, self.ilt2,
                                                                                all_cases, firm_i, 's2')
        # These two vectors store the probabilities that no firm up to i made an offer to s1 or s2.
        return

    def action_comparison(self, ix_s1, ix_s2, firm_i):
        if firm_i == 0:
            return 1
        else:
            exp1 = ((ix_s1+1)/self.n_grid)*self.firmOfferProbs_s1[firm_i-1, ix_s1, ix_s2]
            exp2 = ((ix_s2+1)/self.n_grid)*self.firmOfferProbs_s2[firm_i-1, ix_s1, ix_s2]
            if exp1 >= exp2:
                return 1  # When action_comparison has value 1, make offer to screen 1
            else:
                return 0  # Else, make offer to screen 2

    def action_matrix(self, firm_i):
        act_comp_vectorized = np.vectorize(self.action_comparison)
        self.firmOfferActions[firm_i, self.ilt1, self.ilt2] = act_comp_vectorized(self.ilt1, self.ilt2, firm_i)
        # This means that firmOfferActions[firm,s1,s2] is 1 iff firm makes offer to s1 (and 0 instead).
        return

    # ################################## Functions for Plots #######################################
    def firm_value_interim(self, ix_s1, ix_s2, firm_i):
        if firm_i == 0:
            return (ix_s1+1)/self.n_grid
        else:
            return self.action_comparison(ix_s1, ix_s2, firm_i) * ((ix_s1 + 1) / self.n_grid) * \
                    self.firmOfferProbs_s1[firm_i - 1, ix_s1, ix_s2] + \
                    (1 - self.action_comparison(ix_s1, ix_s2, firm_i)) * \
                    ((ix_s2 + 1) / self.n_grid) * self.firmOfferProbs_s2[firm_i - 1, ix_s1, ix_s2]

    def firm_inner_int(self, ix_s1, firm_i):
        intsum = 0
        for m in range(ix_s1+1):
            intsum += self.firm_value_interim(ix_s1, m, firm_i)
        intsum = intsum/ix_s1
        return intsum

    def firm_value_position(self, firm_i):
        outerintsum = 0
        for n in range(1, self.n_grid):
            outerintsum += 2*n/self.n_grid * self.firm_inner_int(n, firm_i)
        outerintsum = outerintsum/(self.n_grid - 1)
        return outerintsum

    def firm_value_ex_ante(self, q):
        a = 0
        for j in range(1, self.n_firms + 1, 1):
            a += (1 - q) ** (j - 1) * q ** (self.n_firms - j) * self.binom(self.n_firms - 1, j - 1) *\
                 self.firm_value_position(j-1)
        return a

    def match_value_workers(self, s):
        """
        :param s: Skill of the worker (between 0 and self.n_grid) 
        :return: Expected match value when firms are using the optimal strategy
        """  # Function to be corrected!! (Or check that it is correct) #################
        sum = (self.n_firms/(self.n_firms + 1)) * 2/self.n_workers * s/(self.n_grid-1)
        for i in range(1, self.n_firms):
            count = 0  # For each firm, count is supposed to calculate the probability that firm i makes an offer to
            # the worker while all the better firms did not, depending on the second screen of firm i
            for smaller in range(0, s):
                if self.firmOfferActions[i, s, smaller] == 1:
                    count += self.firmOfferProbs_s1[i-1, s, smaller]/(self.n_grid-1)
            for greater in range(s, self.n_grid):
                if self.firmOfferActions[i, greater, s] == 2:
                    count += self.firmOfferProbs_s2[i-1, greater, s]/(self.n_grid-1)
            sum += (self.n_firms-i)/(self.n_firms + 1)*2/self.n_workers*count
        return sum

    def plotfirmvalue(self):
        t = np.arange(0, 1, .01)
        plt.plot(t, self.firm_value_ex_ante(t), 'r-')
        plt.title('Expected match value of a firm with {} firms and {} workers'.format(self.n_firms, self.m))
        # plt.legend(loc="upper right")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("quality")
        plt.ylabel("Expected Match value")
        plt.show()
        return

    def plotworkervalue(self):
        t = np.arange(0, self.n_grid, 1)
        s = []
        for i in t:
            s.append(self.match_value_workers(i))
        plt.plot(t, s, 'r-')
        plt.title('Expected match value of a worker with {} firms and {} workers'.format(self.n_firms, self.m))
        # plt.legend(loc="upper right")
        plt.xlim([0, self.n_grid])
        plt.ylim([0, 1])
        plt.xlabel("Skill level")
        plt.ylabel("Expected Match value")
        plt.show()
        return

    def plot2(self, actions):
        res = np.zeros((self.n_grid, self.n_grid, 3))
        for i, j in itertools.product(range(self.n_grid), range(self.n_grid)):
            if j > i:
                value = [1, 1, 1]
            elif actions[self.n_firms-1, i, j]:
                value = [0.81, 0.81, 0.81]
            elif not actions[self.n_firms-1, i, j]:
                value = [0.27, 0.27, 0.27]
            else:
                value = [1, 0, 0]

            res[j, i, :] = value

        legend_elements = [Line2D([0], [0], color=[0.81, 0.81, 0.81], lw=4, label='Offer to first screen'),
                           Line2D([0], [0], color=[0.27, 0.27, 0.27], lw=4, label='Offer to second screen')]

        plt.figure(dpi=100)
        plt.title('Firm {} Decision, M={}'.format(self.n_firms, self.m))
        plt.xlabel("Skill level first screen")
        plt.ylabel("Skill level second screen")
        plt.imshow(res, origin='lower', extent=(0, 1, 0, 1))
        plt.legend(handles=legend_elements, loc='upper left')
        plt.savefig('Firm {} Decision, M={}.pdf'.format(self.n_firms, self.m), pad_inches=0.01)
        plt.show()
        return

    def calculate(self):
        """
        Calculate expected values for all firms
        This is the main function to run
        """
        # build cases
        match_cases = MatchingCases(self.n_firms, self.m)
        # breakpoint()
        all_cases = match_cases.case_builder()
        # print(all_cases)
        for firm in range(1, self.n_firms):
            # calculate offer probs
            self.prob_no_offer(all_cases, firm-1)
            # calculate actions matrix
            self.action_matrix(firm)  # When this is run for every firm, firmOfferActions is a vector of
            # decision matrices for each firm (given screens s1 and s2)
        # self.plot2(self.firmOfferActions)  # This is needed to create Figure 1
        # self.plotfirmvalue()
        # self.plotworkervalue()
        return

    # ######################## Functions to calculate match values under myopic strategies ############################
    def firmval1(self, s, n, k):
        """ s=skill of top-screened worker, n=number of better firms in my period, k=number of
         workers in my period
        """
        if n == 0:
            return s
        elif k <= 2:
            return 0
        else:
            return s*(1-(1+(k-2)*s)/self.binom(k, 2))**n

    def firmval2(self, n, k):
        if k == 0:
            return 0
        elif k == 1:
            return integrate.quad(lambda x: self.firmval1(x, n, k), 0, 1)[0]
        else:
            return integrate.quad(lambda x: self.firmval1(x, n, k)*2*x, 0, 1)[0]

    def firmval3(self, m, k, q):
        """
        :param m: Total number of Firms
        :param q: Quality of the firm
        :param k: Total number of Workers
        :return: Expected match value of the firm given m,k and q (myopic strategy used by all firms)
        """
        value = 0
        for i in range(m):
            value += self.firmval2(i, k)*(1-q)**i*q**(m-1-i)*self.binom(m-1, i)
        return value

    def assortativefirmval(self, q):
        """
        :param q: Quality of the firm
        :return: expected value from assortative matching
        """
        value = 0
        for i in range(self.n_workers - self.n_firms + 1, self.n_workers + 1, 1):
            match = i/(self.n_workers + 1)
            betterfirms = self.n_workers - i
            value += match * (1-q)**betterfirms * q**(self.n_firms-betterfirms-1) * self.binom(self.n_firms-1, betterfirms)
        return value

    def workerprob(self, n, k):
        """
        :param n: number of other workers in my period
        :param k: number of worse workers in my period
        :return: probability that one firm in this period has me as their top screened worker
        """
        if n == 0:
            return 1
        elif n >= 1 and k == 0:
            return 0
        elif n == 1 and k == 1:
            return 1
        else:
            return k/self.binom(n+1,2)

    def workerval1(self, n, k, N):
        """
        :param n: Total number of workers
        :param k: Total number of worse workers
        :param N: Total number of firms
        :return: expected matchvalue of a worker, depending on n, k and N
        """
        matchvalue = 0
        for i in range(N+1):
            matchvalue += i/(i+1) * self.workerprob(n-1, k)**i * (1 - self.workerprob(n-1, k))**(N-i) * self.binom(N, i)
        return matchvalue

    def workerval2(self, N, n, s):
        """
        :param n: Total number of workers
        :param N: Total number of firms
        :param s: my skill level as a worker
        :return: expected matchvalue of a worker, depending on n, s and N (for myopic strategy of the firms)
        """
        matchvalue = 0
        for j in range(n):
            matchvalue += s**j * (1-s)**(n-1-j) * self.binom(n-1, j) * self.workerval1(n, j, N)
        return matchvalue

    def assortativeworkerval(self, s):
        """
        :param s: Skill of the worker
        :return: expected worker value from assortative matching
        """
        value = 0
        for i in range(1, self.n_firms + 1, 1):
            match = i/(self.n_firms + 1)
            betterworkers = self.n_firms - i
            value += match * (1-s)**betterworkers * s**(self.n_workers - betterworkers - 1) * \
                self.binom(self.n_workers - 1, betterworkers)
        return value


if __name__ == '__main__':
    start_time = time.time()
    model = ExpValues(n_grid=2000, n_firms=2, n_workers=4)
    model.calculate()

    t = np.arange(0, model.n_grid, 1)
    t2 = np.arange(0.0000001, 1, 1 / model.n_grid)

    a1 = []  # designed for assortative matchvalues for firms
    w1 = []  # designed for assortative matchvalues for workers
    b1 = []  # designed for firm values under myopic strategy
    c1 = []  # designed for worker values under equilibrium strategy
    s1 = []  # firm value under equilibrium strategy
    r1 = []  # Relative difference of BNE over Myopic strategy
    for i in t2:
        # s1.append(model.firm_value_ex_ante(i))
        # a1.append(model.assortativefirmval(i))
        # b1.append(model.firmval3(model.n_firms, model.n_workers, i))
        # w1.append(model.assortativeworkerval(i))
        r1.append((model.firm_value_ex_ante(i)-model.firmval3(model.n_firms, model.n_workers, i)) /
                  model.firmval3(model.n_firms, model.n_workers, i))
    # for j in t:
    #     c1.append(model.match_value_workers(j))
    print("--- Runtime %s seconds ---(Model 1 computed)" % int(time.time() - start_time))
    model2 = ExpValues(n_grid=2000, n_firms=4, n_workers=8)
    model2.calculate()

    a2 = []  # designed for assortative matchvalues for firms
    w2 = []  # designed for assortative matchvalues for workers
    b2 = []  # designed for firm values under myopic strategy
    c2 = []  # designed for worker values under equilibrium strategy
    s2 = []  # firm value under equilibrium strategy
    r2 = []  # Relative difference of BNE over Myopic strategy
    for i in t2:
        # s2.append(model2.firm_value_ex_ante(i))
        # a2.append(model2.assortativefirmval(i))
        # b2.append(model2.firmval3(model2.n_firms, model2.n_workers, i))
        # w2.append(model2.assortativeworkerval(i))
        r2.append((model2.firm_value_ex_ante(i) - model2.firmval3(model2.n_firms, model2.n_workers, i)) /
                  model2.firmval3(model2.n_firms, model2.n_workers, i))
    # for j in t:
    #     c2.append(model2.match_value_workers(j))
    print("--- Runtime %s seconds ---(Model 2 computed)" % int(time.time() - start_time))
    # model3 = ExpValues(n_grid=400, n_firms=6, n_workers=12)
    # model3.calculate()
    # a3 = []  # designed for assortative matchvalues for firms
    # w3 = []  # designed for assortative matchvalues for workers
    # s3 = []  # firm value under equilibrium strategy
    # b3 = []  # designed for firm values under myopic strategy
    # c3 = []  # designed for worker values under equilibrium strategy
    # for i in t2:
    #     s3.append(model3.firm_value_ex_ante(i))
    #     w3.append(model3.assortativeworkerval(i))
    # for j in t:
    #     c3.append(model3.match_value_workers(j))
    # print("--- Runtime %s seconds --- (Model 3 computed)" % int(time.time() - start_time))

    # ############ Plots for Firms #############################
    # The Plots below are used in Figures 2, 3, 4 and 5 (Choosing the right values to plot, created above)

    # plt.plot(t2, a1, color='k', linestyle='dashed',
    #          label='{} Firms, {} Workers - Assortative Matching'.format(model.n_firms, model.n_workers))
    # plt.plot(t2, a2, color='0.54', linestyle='dashed',
    #          label='{} Firms, {} Workers - Assortative Matching'.format(model2.n_firms, model2.n_workers))
    # plt.plot(t2, s1, color='k', linestyle='solid',
    #          label='{} Firms, {} Workers'.format(model.n_firms, model.n_workers))
    # plt.plot(t2, b1, 'r--', label='{} Firms, {} Workers - Myopic Strategy'.format(model.n_firms, model.n_workers))
    # plt.plot(t2, s2, color='0.54', linestyle='solid',
    #          label='{} Firms, {} Workers'.format(model2.n_firms, model2.n_workers))
    # plt.plot(t2, s3, color='0.54', linestyle='dashed',
    #          label='{} Firms, {} Workers'.format(model3.n_firms, model3.n_workers))
    # plt.plot(t01, 2/3 * np.exp(-2*(1-t01)), color='0.81', linestyle='solid', label='$2/3 e^{-2(1-q)}$')
    # plt.plot(t2, b3, color='0.54', linestyle='dashed',
    #        label='{} Firms, {} Workers - Myopic Strategy'.format(model2.n_firms, model2.n_workers))
    # # plt.plot(t2, c3, 'b-', label='{} Firms, {} Workers'.format(model3.n_firms, model3.n_workers))
    # plt.legend(loc="upper left")
    # plt.xlim([0, 1])
    # plt.ylim([0.3, 1])
    # plt.xlabel("Firm Quality")
    # plt.ylabel("Expected Match Value")
    # plt.savefig('AssortativeFgray6F.pdf')
    # plt.savefig('MyopicF.pdf')
    # plt.savefig('FirmValueComparison6Fgray.pdf')
    # plt.show()
    # plt.close()

# ################# Plots for Workers #################################################

    # plt.plot(t2, w1, color='k', linestyle='dashed',
    #          label='{} Firms, {} Workers - Assortative Matching'.format(model.n_firms, model.n_workers))
    # plt.plot(t2, w2, color='0.54', linestyle='dashed',
    #          label='{} Firms, {} Workers - Assortative Matching'.format(model2.n_firms, model2.n_workers))
    # plt.plot(t2, c1, color='k', linestyle='solid',
    #          label='{} Firms, {} Workers'.format(model.n_firms, model.n_workers))
    # # plt.plot(t2, b1, 'r--', label='{} Firms, {} Workers - Myopic Strategy'.format(model.n_firms, model.n_workers))
    # plt.plot(t2, c2, color='0.54', linestyle='solid',
    #          label='{} Firms, {} Workers'.format(model2.n_firms, model2.n_workers))
    # plt.plot(t2, c3, color='0.54', linestyle='dashed',
    #          label='{} Firms, {} Workers'.format(model3.n_firms, model3.n_workers))
    # # plt.plot(t2, b2, 'g--', label='{} Firms, {} Workers - Myopic Strategy'.format(model2.n_firms, model2.n_workers))
    # # plt.plot(t2, c3, 'b-', label='{} Firms, {} Workers'.format(model3.n_firms, model3.n_workers))
    # plt.legend(loc="upper left")
    # plt.xlim([0, 1])
    # plt.ylim([0, 0.9])
    # plt.xlabel("Worker Skill")
    # plt.ylabel("Expected Match Value")
    # plt.savefig('AssortativeWgray6F.pdf')
    # plt.savefig('MyopicW.pdf')
    # plt.savefig('WorkerValueComparison6Fgray.pdf')
    # plt.close()

    # ############ Subplots for workers (low or high skill) ####################

    # t03 = np.arange(0.1, 0.3, 1/model.n_grid)  # for the two firm case
    # t031 = np.arange(80, 240, 1)  # for the two firm case, model.n_grid=800
    # t3 = np.arange(0.1, 0.3, 1/model2.n_grid)
    # t31 = np.arange(40, 120, 1)  # for model2.n_grid=400
    # d1 = []  # designed for low skill worker values
    # d2 = []
    # d3 = []
    # for j in t031:
    #     d1.append(model.match_value_workers(j))
    # for j in t31:
    #     d2.append(model2.match_value_workers(j))
    #     d3.append(model3.match_value_workers(j))
    # t04 = np.arange(0.8, 1, 1/model.n_grid)
    # t041 = np.arange(640, model.n_grid, 1)  # for model.n_grid=800
    # t4 = np.arange(0.8, 1, 1/model2.n_grid)
    # t41 = np.arange(320, model2.n_grid, 1)  # for model2.n_grid=400
    # e1 = []  # designed for high skill worker values
    # e2 = []
    # e3 = []
    # for j in t041:
    #     e1.append(model.match_value_workers(j))
    # for j in t41:
    #     e2.append(model2.match_value_workers(j))
    #     e3.append(model3.match_value_workers(j))

    # plt.plot(t03, d1, color='k', linestyle='solid',
    #         label='{} Firms, {} Workers'.format(model.n_firms, model.n_workers))
    # plt.plot(t3, d2, color='0.27', linestyle='dotted',
    #          label='{} Firms, {} Workers'.format(model2.n_firms, model2.n_workers))
    # plt.plot(t3, d3, color='0.54', linestyle='dashed',
    #          label='{} Firms, {} Workers'.format(model3.n_firms, model3.n_workers))
    # plt.legend(loc="upper left")
    # plt.xlabel("Worker Skill")
    # plt.ylabel("Expected Match Value")
    # plt.xlim([0.1, 0.3])
    # plt.ylim([0.04, 0.15])
    # plt.savefig('WorkerValueComparison6Fgraylow.pdf')
    # plt.close()

    # plt.plot(t04, e1, color='k', linestyle='solid',
    #          label='{} Firms, {} Workers'.format(model.n_firms, model.n_workers))
    # plt.plot(t4, e2, color='0.27', linestyle='dotted',
    #          label='{} Firms, {} Workers'.format(model2.n_firms, model2.n_workers))
    # plt.plot(t4, e3, color='0.54', linestyle='dashed',
    #          label='{} Firms, {} Workers'.format(model3.n_firms, model3.n_workers))
    # plt.legend(loc="upper left")
    # plt.xlabel("Worker Skill")
    # plt.ylabel("Expected Match Value")
    # plt.xlim([0.8, 1])
    # plt.ylim([0.3, 0.4])
    # plt.savefig('WorkerValueComparison6Fgrayhigh.pdf')
    # plt.close()

# #################### Plot for comparison of Myopic and BNE strategy ###########################

    plt.plot(t2, r1, color='k', linestyle='solid',
             label='{} Firms, {} Workers'.format(model.n_firms, model.n_workers))
    plt.plot(t2, r2, color='0.54', linestyle='solid',
             label='{} Firms, {} Workers'.format(model2.n_firms, model2.n_workers))
    plt.legend(loc="upper right")
    plt.xlim([0, 1])
    # plt.ylim([0, 0.06])
    plt.xlabel("Firm Quality")
    plt.ylabel("Relative Difference")
    plt.savefig('ComparisonMyopicBNE-2-4.pdf')
    plt.close()
