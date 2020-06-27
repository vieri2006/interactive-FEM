import matplotlib
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
IPython.display.set_matplotlib_formats("svg")

matplotlib.style.use("dark_background")

init_printing(use_latex=True)

# # Symbols
x = var("x", real=True)
L = Symbol("L", real=True)
T = Function("T")(x)
w = Function("w")(x)
q0 = Symbol("q_0", real=True)
qL = Symbol("q_L", real=True)
k = Symbol("k", real=True, positive=True)
Q = Symbol("Q", real=True)


class FEM(object):
    """FEM FOR 1 Dimensional Heat Transfer with specific BC: 
    konstan q pada 0 dan konstan T pada L"""

    def __init__(self, orde, Nelements, Lnum, T_Lnum, q0num, knum, Qnum, ):
        """FEM Setup:
        Params:
        orde : int
            Orde Shape function, dari 1 sampai 3
        Nelements : int
            Karena keterbatasan symbolic computation: 
            - untuk linear, maks = 5
            - untuk quadratic, maks = 6, harus kelipatan 2
            - untuk cubic, maks = 6, harus kelipatan 3
        L : float
            Panjang batang
        T_Lnum : float
            T pada x = L
        q0 : float
            fluks kalor pada x = 0
        k : float
            Konduktivitas bahan;
            kalau mau ubah ke k sbg fungsi x, cek method changeK
        Q : float
            Generasi kalor bahan;
            kalau mau ubah ke Q sbg fungsi x, cek method changeQ
        """
        # # Design Specification
        self.T_Lnum = T_Lnum
        self.Lnum = Lnum
        self.knum = knum
        self.Qnum = Qnum
        self.q0num = q0num
        self.orde = orde
        self.Nelements = Nelements
        self.Npoints = self.Nelements + 1

        self.isKchange = False

        assert isinstance(
            self.orde, int) and self.orde >= 1 and self.orde <= 3, "orde harus integer antara 1 dan 3"

        assert isinstance(self.Nelements, int), "Nelements harus integer"
        assert self.Nelements % self.orde == 0, "Harus kelipatan {} sesuai orde".format(
            self.orde)

        # # Discretization

        self.xnodes = []
        for i in range(self.Npoints):
            self.xnodes += [L / self.Nelements * i]

        self.h = [simplify(self.xnodes[i+1]-self.xnodes[i])
                  for i in range(self.Nelements)]

        self.T_i = []
        for i in range(self.Npoints):
            var_sym = "T_" + str(i)
            self.T_i.append(Symbol(var_sym, real=True))

        # # For future sympy subs
        self.num_dict = {L: self.Lnum, k: self.knum,
                         Q: self.Qnum, q0: self.q0num, self.T_i[-1]: self.T_Lnum}

        _initN(self)

    def changeNode(self, index, value):
        """To change node points
        Params: 
        index: int
            Index node yang diganti
        value: float
            Nilai node yang baru 
        """
        assert 0 < index < self.Nelements and isinstance(
            index, int), "Index must be between 1 and {} ".format(self.Nelements-1)
        assert self.xnodes[index - 1].subs(self.num_dict) < value < self.xnodes[index + 1].subs(
            self.num_dict), "New node must be between {} and {})".format(self.xnodes[index - 1], self.xnodes[index + 1])

        self.xnodes[index] = value
        self.h = [simplify(self.xnodes[i+1]-self.xnodes[i])
                  for i in range(self.Nelements)]

        _initN(self)

    def changeQ(self, Qfunc):
        """To change Heat Generation as a function of x
        Params: 
        Qfunc: str
            String of function to be parsed as the heat generation
        """
        isinstance(Qfunc, str), "Parameter must be string, try using \"\""
        self.Qfunc = parse_expr(Qfunc)
        self.Qsymbols = list(self.Qfunc.free_symbols)
        assert len(self.Qsymbols) <= 1, "Must only consist of x symbols"
        x_ = self.Qsymbols[0]
        QNodes = [self.Qfunc.evalf(
            subs={x_: i.subs(self.num_dict)}) for i in self.xnodes]
        self.Qnum = sum([QNodes[i] * self.N[i] for i in range(self.Npoints)])

        NptsPlot = 41
        xplt = np.linspace(0.0, self.Lnum, NptsPlot)
        xplt = [float(ip) for ip in xplt]
        xplt[0] = int(0)
        plt.clf()
        yplt = [self.Qnum.subs(self.num_dict).subs(x, ip) for ip in xplt]
        plt.plot(xplt, yplt, label="Q generated")
        plt.legend()

    def changeK(self, Kfunc):
        """To change Conductivity as a function of x
        Params: 
        Kfunc: str
            String of function to be parsed as the conductivity
        """
        self.isKchange = True

        self.Kfunc = parse_expr(Kfunc)
        self.Ksymbols = list(self.Kfunc.free_symbols)
        assert len(self.Ksymbols) <= 1, "Must only consist of x symbols"
        x_ = self.Ksymbols[0]

        isinstance(Kfunc, str), "Parameter must be string, try using \"\""
        self.Kfunc = parse_expr(Kfunc)

        KNodes = [self.Kfunc.evalf(
            subs={x_: i.subs(self.num_dict)}) for i in self.xnodes]
        self.knum = sum([KNodes[i] * self.N[i] for i in range(self.Npoints)])

        NptsPlot = 41
        xplt = np.linspace(0.0, self.Lnum, NptsPlot)
        xplt = [float(ip) for ip in xplt]
        xplt[0] = int(0)
        plt.clf()
        yplt = [self.knum.subs(self.num_dict).subs(x, ip) for ip in xplt]
        plt.plot(xplt, yplt, label="Conductivity")
        plt.legend()

    def evaluate(self):
        # # T Expansion
        self.T_expansion = 0
        for i in range(self.Npoints):
            self.T_expansion += self.T_i[i] * self.N[i]

        # # Derivation of linear systems
        eq = []
        self.term1 = []
        self.term2 = []
        self.term3 = []

        # ## Defining each self.terms
        expr1 = k * Derivative(w, x) * Derivative(T, x)
        expr1 = expr1.subs(T, self.T_expansion)
        expr2 = Integral(Q*w, (x, 0, L))
        expr3 = k*w*Derivative(T, x)

        # ## Evaluating each self.terms
        # ### Stiffness / Diffusion self.term
        print("This will be long\nEvaluating Stiffness Matrix")
        for i in range(0, self.Npoints):
            print("\t self.term: ", i)
            expr1s = expr1.subs({w: self.N[i], k: self.knum})
            expr1ss = simplify(expr1s.doit())
            I1 = Integral(expr1ss, (x, 0, L))
            self.term1.append(I1.doit())

        # ### Source self.term
        print("Evaluating Source Term")
        for i in range(0, self.Npoints):
            print("\t self.term: ", i)
            self.term2.append(expr2.subs({w: self.N[i], Q: self.Qnum}).doit())

        # ### Boundary self.terms
        print("Evaluating Boundary Term")
        for i in range(0, self.Npoints):
            print("\t self.term: ", i)
            self.term3_x0 = expr3.subs({w: self.N[i], x: 0, L: self.Lnum})
            self.term3_xL = expr3.subs(
                {w: self.N[i], x: self.Lnum, L: self.Lnum})
            self.term3.append(self.term3_xL - self.term3_x0)

        # ### Overall contributions for the first weight function
        for i in range(0, self.Npoints):
            eq.append(self.term1[i] - self.term2[i] - self.term3[i])

        eq[0] = eq[0].subs({k*self.term3[0].args[2]: -q0})
        eq[-1] = eq[-1].subs({k*self.term3_xL.args[1]: -qL})

        # ### Some modification if K varies
        """ DITUNDA KARENA HASIL ANEH
        if self.isKchange == True:
            print("Conductivity is customized: Modifying a bit")
            self.term4 = []
            expr4 = -w*Derivative(self.knum, x)*Derivative(T, x)
            expr4 = expr4.subs(T, self.T_expansion)

            for i in range(0, self.Npoints):
                print(i, "/", self.Nelements)
                expr4s = expr4.subs({w: self.N[i]})
                expr4ss = simplify(expr4s.doit())
                I4 = Integral(expr4ss, (x, 0, L))
                self.term4.append(I4.doit())

            for i in range(0, self.Npoints):
                eq[i] += self.term4[i]
        """

        # # System of linear equations
        eq = [eqs.subs(self.num_dict) for eqs in eq[0:self.Npoints]]

        print("Solving for T and qL")
        self.sols_T = linsolve(eq[0:-1], self.T_i[0:-1])
        self.sols_T = list(list(self.sols_T)[0]) + [self.T_Lnum]
        self.sols_T_dict = list(zip(self.T_i[0:-1], self.sols_T[0:-1]))

        self.sols_qL = solve(eq[-1], qL)[0]
        self.sols_qL = self.sols_qL.subs(self.num_dict).subs(self.sols_T_dict)


def plotShapeFunction(FEM_object):
    """Plot the shape function of the FEM objects"""
    N_plot = [n.subs(FEM_object.num_dict).evalf() for n in FEM_object.N]

    NptsPlot = 91
    xplt = np.linspace(0.0, FEM_object.Lnum, NptsPlot)
    xplt = [float(ip) for ip in xplt]
    xplt[0] = int(0)
    yplt = []
    plt.clf()
    for i in range(len(N_plot)):
        yplt.append([N_plot[i].subs(x, ip) for ip in xplt])
        plt.plot(xplt, yplt[i], label="N_" + str(i))
    plt.legend()


def plotTExpansion(FEM_object):
    """Plot the weighted shape functions of the FEM objects"""
    N_plot = [n.subs(FEM_object.num_dict).evalf() for n in FEM_object.N]

    NptsPlot = 91
    xplt = np.linspace(0.0, FEM_object.Lnum, NptsPlot)
    xplt = [float(ip) for ip in xplt]
    xplt[0] = int(0)
    yplt = []
    plt.clf()
    for i in range(len(N_plot)):
        yplt.append([FEM_object.sols_T[i] * N_plot[i].subs(x, ip)
                     for ip in xplt])
        plt.plot(xplt, yplt[i], label="N_" + str(i))
    plt.legend()


def plotT(FEM_object):
    """Plot the Temperature distribution of the FEM objects"""
    T_plot = FEM_object.T_expansion.subs(
        FEM_object.num_dict).subs(FEM_object.sols_T_dict).evalf()

    NptsPlot = 91
    xplt = np.linspace(0.0, FEM_object.Lnum, NptsPlot)
    xplt = [float(ip) for ip in xplt]
    xplt[0] = int(0)
    yplt = [T_plot.subs(x, ip) for ip in xplt]
    plt.clf()
    plt.plot(xplt, yplt, label="Temperature")
    plt.legend()


def _initN(FEM_object):
    if FEM_object.orde == 3:
        FEM_object.N = [0]
        for i in range(0, FEM_object.Nelements, 3):
            cond = (x >= FEM_object.xnodes[i]) & (x < FEM_object.xnodes[i+3])
            f1 = ((x - FEM_object.xnodes[i+1]) * (x - FEM_object.xnodes[i+2]) * (x - FEM_object.xnodes[i+3])
                  / (FEM_object.xnodes[i] - FEM_object.xnodes[i+1]) / (FEM_object.xnodes[i] - FEM_object.xnodes[i+2]) / (FEM_object.xnodes[i] - FEM_object.xnodes[i+3]))
            FEM_object.N[-1] += simplify(Piecewise((f1, cond), (0, True)))
            f2 = ((x - FEM_object.xnodes[i]) * (x - FEM_object.xnodes[i+2]) * (x - FEM_object.xnodes[i+3])
                  / (FEM_object.xnodes[i+1] - FEM_object.xnodes[i]) / (FEM_object.xnodes[i+1] - FEM_object.xnodes[i+2]) / (FEM_object.xnodes[i+1] - FEM_object.xnodes[i+3]))
            FEM_object.N.append(simplify(Piecewise((f2, cond), (0, True))))
            f3 = ((x - FEM_object.xnodes[i]) * (x - FEM_object.xnodes[i+1]) * (x - FEM_object.xnodes[i+3])
                  / (FEM_object.xnodes[i+2] - FEM_object.xnodes[i]) / (FEM_object.xnodes[i+2] - FEM_object.xnodes[i+1]) / (FEM_object.xnodes[i+2] - FEM_object.xnodes[i+3]))
            FEM_object.N.append(simplify(Piecewise((f3, cond), (0, True))))
            f4 = ((x - FEM_object.xnodes[i]) * (x - FEM_object.xnodes[i+1]) * (x - FEM_object.xnodes[i+2])
                  / (FEM_object.xnodes[i+3] - FEM_object.xnodes[i]) / (FEM_object.xnodes[i+3] - FEM_object.xnodes[i+1]) / (FEM_object.xnodes[i+3] - FEM_object.xnodes[i+2]))
            FEM_object.N.append(simplify(Piecewise((f4, cond), (0, True))))

        cond = (x >= FEM_object.xnodes[-4]) & (x <= FEM_object.xnodes[-1])
        f4 = ((x - FEM_object.xnodes[i]) * (x - FEM_object.xnodes[i+1]) * (x - FEM_object.xnodes[i+2])
              / (FEM_object.xnodes[i+3] - FEM_object.xnodes[i]) / (FEM_object.xnodes[i+3] - FEM_object.xnodes[i+1]) / (FEM_object.xnodes[i+3] - FEM_object.xnodes[i+2]))
        FEM_object.N[-1] = (Piecewise((f4, cond), (0, True)))

    if FEM_object.orde == 2:
        FEM_object.N = [0]
        for i in range(0, FEM_object.Nelements, 2):
            cond = (x >= FEM_object.xnodes[i]) & (x < FEM_object.xnodes[i+2])
            f1 = (FEM_object.xnodes[i+1] - x) * (FEM_object.xnodes[i+2] -
                                                 x) / FEM_object.h[i] / (FEM_object.h[i] + FEM_object.h[i+1])
            FEM_object.N[-1] += simplify(Piecewise((f1, cond), (0, True)))
            f2 = (FEM_object.xnodes[i] - x) * (FEM_object.xnodes[i +
                                                                 2] - x) / (-FEM_object.h[i]) / FEM_object.h[i+1]
            FEM_object.N.append(simplify(Piecewise((f2, cond), (0, True))))
            f3 = (FEM_object.xnodes[i] - x) * (FEM_object.xnodes[i+1] - x) / \
                FEM_object.h[i+1] / (FEM_object.h[i] + FEM_object.h[i+1])
            FEM_object.N.append(simplify(Piecewise((f3, cond), (0, True))))

        cond = (x >= FEM_object.xnodes[-3]) & (x <= FEM_object.xnodes[-1])
        f3 = ((FEM_object.xnodes[-3] - x) * (FEM_object.xnodes[-2] - x)
              / (FEM_object.xnodes[-1] - FEM_object.xnodes[-3]) / (FEM_object.xnodes[-1] - FEM_object.xnodes[-2]))
        FEM_object.N[-1] = (simplify(Piecewise((f3, cond), (0, True))))

    if FEM_object.orde == 1:
        FEM_object.N = []
        cond_lbc = (x >= FEM_object.xnodes[0]) & (x <= FEM_object.xnodes[1])
        f1 = simplify((FEM_object.xnodes[1] - x)/FEM_object.h[0])
        FEM_object.N.append(Piecewise((f1, cond_lbc), (0, True)))

        for i in range(1, FEM_object.Nelements):
            cond1 = (x >= FEM_object.xnodes[i-1]) & (x <= FEM_object.xnodes[i])
            f1 = (x - FEM_object.xnodes[i-1])/FEM_object.h[i-1]

            cond2 = (x >= FEM_object.xnodes[i]) & (x <= FEM_object.xnodes[i+1])
            f2 = (FEM_object.xnodes[i+1] - x)/FEM_object.h[i]

            FEM_object.N.append(Piecewise((f1, cond1), (f2, cond2), (0, True)))

        cond_rbc = (x >= FEM_object.xnodes[FEM_object.Nelements-1]
                    ) & (x <= FEM_object.xnodes[FEM_object.Nelements])
        f1 = (x - FEM_object.xnodes[FEM_object.Nelements-1]
              )/FEM_object.h[FEM_object.Nelements-1]
        FEM_object.N.append(Piecewise((f1, cond_rbc), (0, True)))
