# les importations 
import base64
from io import BytesIO
from flask import Flask , render_template , url_for ,request,Response
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from math import *

#Flask 
app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("Home.html")

@app.route("/commencer")
def commencer():
    return render_template("commencer.html")

@app.route("/A_propos")
def A_propos():
    return render_template("A propos.html")

@app.route("/service")
def service():
    return render_template("service.html")

@app.route("/commencer_Lagrange")
def commencer_Lagrange():
    return render_template("commencer_Lagrange.html")

@app.route("/commencer_Newton")
def commencer_Newton():
    return render_template("commencer_Newton.html")

@app.route("/commencer_Hermite")
def commencer_Hermite():
    return render_template("commencer_Hermite.html")
@app.route("/fonc")
def fonc():
    return render_template("fonc.html")

#@app.route('/', methods=['GET'])
#def graph4():
#    return render_template('Home.html')

global subject
global name 
name = "(12;5.25),(10.01;3),(5;0)"
global email
global subject1 
subject1=" (1/(0.5*(2*3.1415926)**0.5)*e((0-0.5)*(x-1)**2/0.5**2))"
global subject2
global F
global L
global P
global deg
global testing
testing=1 
global warning 
warning=""

@app.route("/login",methods=['get', 'post'] )   # getvalue de la page /plott 
def getvalue():
    #degree=subject
    # xi = name yi= email 
    #saisir fonction = subjetct1 
    #erreur = subject2
    try:
        subject = request.form['subject']    # ces variables sont de type str 
        name = request.form['name']
        subject1 = request.form['subject1']
        subject3 = request.form['subject3']
        
        #######"fonctions "
        def saisie_facultatif(f) :
            warning=''
            """éviter les espaces"""
            """éviter les puissances négatives"""
            """les constantes pi et e ne sont pas définie"""
            """la variable que x ou X"""
            """-x n'est pas accepté --> 0-x"""
            
            if f == "" :
                return "",warning
            test = 1

            #nombre de parenthèse ouvrant égal nombre de parenthèse fermant 
            openn = []
            close = []
            for i in range ( len(f) ) :
                if f[i] == "(" :
                    openn.append(i)
                elif f[i] == ")" :
                    close.append(i)
            if len(openn) != len(close):
                warning+='il manque un parenthèse\n'
                
            #vérifier la forme de l'éxpression
            if '***' in f:
                warning+='vérifier les opérants * et **\n'
                
            i = 0
            while i < len(f)-1:
                if (f[i:i+3] =='log') :
                    #verification de la base de log
                    par = f[i:].find('(') + i
                    try :
                        float(f[i+3 : par])
                    except:
                        test = 0
                        #yield i
                        warning+='verifier la fonction log :: position '+str(i)+'\n'
                        break
                    #à se niveau log est suivi d'une base la forme est correcte exp log10
                    #verivication de l'expression entre les parenthéses de log exp log10(x**2)
                    if f[par + 1] in [')','+','-','*','/'] :
                        #exp log2() ou log2(-x) {la forme correcte log2(0-x)}
                        test = 0 
                        #yield i
                        warning+='verifier la fonction log :: position '+str(i)+'\n'
                        break
                    i += par + 1
                    #print(i)
                    continue
                if (f[i].isnumeric()) :
                    #n est numérique ce qui est vraie: 
                    # n) , n+ , n- , n* , n/ , ou suivie d'un autre chiffre n1 exp 21 , ou virgule 5.
                    # (n , +n , -n , *n , /n , ou précédé d'un autre chiffre 2n exp 22 , ou virgule .2
                    if (f[i + 1] not in [')','+','-','*','/','0', '1','2','3', '4', '5', '6', '7', '8', '9','.']) or (f[i - 1] not in ['(','+','-','*','/','0', '1','2','3', '4', '5', '6', '7', '8', '9','.']) :
                        test = 0 
                        #yield i
                        warning +='un opérant ou des parenthèses manquent :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i].lower() == 'x') :
                    #comme le cas numérique mais on peut avoir 2x ou x2
                    if (f[i + 1] not in [')','+','-','*','/']) or (f[i - 1] not in ['(','+','-','*','/']) :
                        test = 0 
                        #yield i
                        warning+='un opérant ou des parenthèses manquent :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if f[i] == ')' :
                    #les cas corrects: x) , chiffre) , ))
                    # )+ , )- , )* , )/ , )) 
                    if (f[i - 1] not in ['x', 'X', '0', '1','2','3', '4', '5', '6', '7', '8', '9', ')']) or (f[i + 1] not in [')','+','-','*','/']) :
                        test = 0 
                        #yield i 
                        warning+='vérifier les parenthèses :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i] == '*'):
                    # x* , chiffre* , )* , **
                    # les cas à éviter : *) , *+ , *- , */ 
                    if (f[i - 1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')','*']) or (f[i + 1] in [')','+','-','/']):
                        test = 0
                        #yield i 
                        warning+='vérifier les opérants :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i] in ['+','/','-' ] ):
                    # les cas possibles x+ , chiffre+ , )+ de même pour / et -
                    # les cas à éviter : +) , ++ , +- , +/ , +. de même pour / et -
                    if (f[i - 1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')']) or (f[i + 1] in [')','+','-','*','/','.']):
                        test = 0
                        #yield i 
                        warning+= 'vérifier les opérants :: position '+str(i)+'\n'
                        break
                    if (f[i] == '/') and (f[i+1] == '0') and f[i+2] != '.':
                        test = 0
                        #yield i 
                        warning+= 'diviseur de zéro :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                i += 1
            #verifier le dernier element de l'expression
            if f[len(f)-1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')']:
                test = 0
                warning+= "l'expression n'a pas terminé correctement \n"
            #if test == 1 :
            #l'expression verifie toutes les contraites
            return f,warning
                    
        def  calcul_simple (expression, x) :
            warning=""
        
            #expression en fonction de x (et/ou X) sans parenthése ne contient que des opérants parmi {+, -, /, *, **} 
            # exemple "x*5**2+X/2+1"
            expression = expression.replace('X','x')
            expression = expression.replace('x',str(x))

            l = []
            i=1
            while i< len(expression):
                if expression[i] in ['+','-','*','/']:
                    l.append(float(expression[:i]))
                    if expression[i+1] == "*":
                            l.append('**')
                            expression = expression[i+2 :]
                            i=0
                    else:
                        l.append(expression[i])
                        expression = expression[i+1 :]
                        i=0
                i += 1
            l.append(float(expression))
            try :
                #calcul des puissances
                i = 1
                while i < len(l) :
                    if l[i] == '**' :
                        y = l[: i-1] + [float(l[i-1]) ** float(l[i+1])] + l[i+2 : ]
                        l = y
                    else:
                        i += 1
                #calcul des * et /
                i = 1
                while i < len(l) :
                    if l[i] == '*' :
                        y = l[:i-1] + [float(l[i-1]) * float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    if l[i] == '/' :
                        y = l[:i-1] + [float(l[i-1]) / float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    i += 1
                #calcul des + et -
                i = 1
                while i < len(l):
                    if l[i] == '+' :
                        y = l[:i-1] + [float(l[i-1]) + float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    if l[i] == '-' :
                        y = l[:i-1] + [float(l[i-1]) - float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    i += 1
            except :
                testing=0
                warning += "erreur: paramètre multiple \nerreur: F doit être en fonction de x \n "

            return l[0]

        def calcul_avance (fonction, x):
            #fonction "tan" ou "e"
            
            import math as m
            if fonction == 'tan' : return round(m.tan(x),15)
            elif fonction == 'sin' : return round(m.sin(x),15)
            elif fonction == 'cos' : return round(m.cos(x),15)
            elif fonction == 'asin' : return round(m.asin(x),15)
            elif fonction == 'acos' : return round(m.acos(x),15)
            elif fonction == 'atan' : return round(m.atan(x),15)
            elif fonction == 'sinh' : return round(m.sinh(x),15)
            elif fonction == 'cosh' : return round(m.cosh(x),15)
            elif fonction == 'tanh' : return round(m.tanh(x),15)
            elif fonction == 'e' : return round(m.exp(x),15)
            elif fonction == 'ln' : return round(m.log(x),15)
            elif fonction == 'log' : return round(m.log10(x),15)
            elif fonction == 'E' : return round(m.floor(x),15)
            elif fonction == 'factorial' : return round(m.factorial(x),15)
            elif fonction[:3] == 'log' :
                    base = int(fonction[3:])
                    return round(m.log(x, base),15)
            else:
                return "fonction n'existe pas"

        def image_par_fonction(expression,x):
            #expression chaine de caractère "sin(x)+2*(exp(-2x)+5)"
            #y = F(x)
            i=0
            openn = []
            test = True
            while i < len(expression):
                if expression[i] == "(" :
                    openn.append(i)
                    i += 1
                elif expression[i] == ")" :
                    block = expression[openn[-1] +1: i]
                    ch2 = expression[i + 1 :]
                    val_block = calcul_simple (block, x)
                    if expression[openn[-1] - 1].isalpha() or  expression[openn[-1] - 1].isnumeric():
                        j = openn[-1] - 1
                        while expression[j] not in ["+", "-", "/", "*", "**", "("] and j>=0 : 
                            j -= 1
                        fonction = expression[j+1 : openn[-1]]
                        
                        val = calcul_avance(fonction, val_block)
                        ch1 = expression [: j+1]
                    else:
                        val = val_block
                        ch1 = expression[:openn[-1]]
            
                    if type(val) == type("A") :
                        test = False
                        break
                    expression = ch1 + str(val)
                    i = len(expression)
                    expression += ch2
                    openn.pop( len(openn)-1 )
                else: 
                    i += 1
            if test  :    
                return calcul_simple(expression, x)
            return val

        def polynome_lagrange (l, i) :
            xi = l[i]
            lx=l[:i]+l[i+1:]
            denominateur = 1
            for xj in lx :
                denominateur *= xi-xj
            for i in range(len(lx)):
                lx[i] *= 0-1
            
            coef_poly_lag = [1/denominateur, sum(lx)/denominateur]

            for r in range(2,len(lx)):
                ai = 0
                comb = combinaisons(lx, r)
                for c in comb:
                    prod=1
                    for j in range(r):
                        prod *= c[j]
                    ai += prod
                coef_poly_lag.append(ai/denominateur)
            a0 = coef_poly_lag[0]
            for xj in lx:
                a0 *= xj
        
            coef_poly_lag.append(a0)
            
            return coef_poly_lag

        def image_par_interpolation(x,coef_poly_inter):
            # coef_poly_inter est une liste contenant les coefficients du polynome d'interpolation
            n = len(coef_poly_inter) - 1
            y = 0
            for ai in coef_poly_inter:
                y += ai * x**n
                n -= 1
            return y

        #x="(0;0.107981),(0.5;0.483941),(1;0.797885),(1.5;0.483941)"
        def polynome_interpolation(L):
            # L liste contenant les coordonnées des points [(x0,y0), (x1,y1)]
            
            n = len(L)
            lx = []
            ly = []
            for point in L:
                lx.append(point[0]) 
                ly.append(point[1])
            
            # on stock les coefficients du polynome d'interpolation dans la liste L
            L0 = polynome_lagrange(lx, 0)
            PI=[0 for i in range(n)]
            for j in range(n):
                    PI[j] = L0[j] * ly[0]
            for i in range(1,n):
                Li = polynome_lagrange(lx, i)
                for j in range(n):
                    PI[j] += Li[j] * ly[i]
            return PI

        def combinaisons(iterable, r):
            # combinaisons('ABCD', 2) --> AB AC AD BC BD CD
            # combinaisons(range(4), 3) --> 012 013 023 123
            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))
            yield tuple(pool[i] for i in indices)
            while True:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)

        def erreur (L,P,F):
            # renvoie max|F(x)-P(x)|
            #P : [an, an-1, .., a0]
            #F : chaine exemple "exp(x)+sin(x)"
            #L : liste des coordonnées des points
            #lx liste des xi
            if F == "":
                return ""
            lx = []
            for point in L:
                lx.append(point[0])

            emax=0
            posMax=None
            for i in range(len(lx)-1):
                vect = np.linspace(lx[i], lx[i+1], 10)
                
                for x in vect:
                    e = image_par_fonction(F,x)-image_par_interpolation(x, P) 
                    if abs(e)> abs(emax) :
                        emax = e
                        posMax=x
            return (emax,posMax)

        def points(ch):
            if name !="":
                l=ch.split(",")
                point=[]
                for e in l:
                    sep=e.index(";")
                    x=e[1:sep]
                    y=e[sep+1:-1]
                    a=float(x)
                    b=float(y)
                    point.append((a,b))
                return (point)
            else:
                return []

        def expression(P):
            #P liste des coefficient du polynome
            n=len(P)
            ch=""
            for i in range (n):
                if P[i]==0:
                    continue
                elif ch!='':
                    if P[i]>0:
                        ch+="+"
                if n-i-1 == 1 or n-i-1 == 0:
                    if P[i]==1:
                        ch+=" x "*(n-i-1)
                    else:
                        ch+=str(P[i])+" x "*(n-i-1)
                else:
                    if P[i]==1:
                        ch+=" x**"+str(n-1-i)+" "
                    else:
                        ch+=str(P[i])+" x**"+str(n-1-i)+" "
            return ch

        
        L=points(name) 
        if len (L)!=int (subject):
            return render_template("about.html",subject=subject)    # here 

        m=saisie_facultatif(subject1)
        F=m[0]
        warning=m[1]

        #F=" (1/(0.5*(2*3.1415926)**0.5)*e((0-0.5)*(x-1)**2/0.5**2))"
        P=polynome_interpolation(L)  
        if warning !="":
            return render_template("about.html",F=F,warning=warning)
        else:
            T= erreur (L,P,F)
            if T != "":
                emax=T[0]
                posMaxErr=T[1]
            ######""#
            
            fig = Figure(figsize=(10,6))
            axis = fig.subplots()
            L.sort()
            lx = []
            ly = []
            for point in L:
                lx.append(point[0]) 
                ly.append(point[1])
            axis.plot(lx, ly,'x', label="Points")
            x = np.linspace(lx[0], lx[-1], 100)
            Py= []
            Fy= []

            if (P != None):
                for j in x:
                    Py.append(image_par_interpolation(j, P))
                axis.plot(x, Py, label="P(x)")
            if (F != ""):
                for j in x:
                    Fy.append(image_par_fonction(F,j))
                axis.plot(x, Fy, label="F(x)")
                if (posMaxErr != None):
                    a=image_par_fonction(F,posMaxErr)
                    b=image_par_interpolation(posMaxErr, P) #here
                    axis.plot([posMaxErr,posMaxErr],[min(b,a),max(b,a)],'r')
            if subject3!="":
                x=float(subject3)
                apimx=image_par_interpolation(x, P)
                err="NONE"
                if F!="":
                    err=image_par_fonction(F,x)-apimx
            else:
                x="x"
                apimx="NONE"
                if F!="":
                    err=image_par_fonction(F,x)-apimx
            # Save it to a temporary buffer.
            buf = BytesIO()
            fig.savefig(buf, format="png")
            # Embed the result in the html output.
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            ch1 =""
            ch2 =""
            ch3 =""
            expr=expression(P)
            ch1="La liste des points entrées ="+str(L)
            if T != "":
                ch3="Pour minimiser l'erreur ,veuillez ajouter un point à l'abscisse :"+str(posMaxErr)
                ch2="L'erreur maximale ="+str(emax)
            

            return render_template("affichage.html",data=data, ch1=ch1,ch2=ch2,ch3=ch3,expr=expr,x=x,apimx=apimx,err=err)
    except:
        cons="Voici quelques conseils:"
        univar="*verifier que la fonction est univariable"
        predef="*vérifier les noms des fonction prédéfinies"
        cnx="*vérifier la connexion"
        return render_template("about.html",cons=cons,univar=univar,predef=predef,cnx=cnx)

        

        #return f"<img src='data:image/png;base64,{data}'/>"+"<br/>"+str(L)+"<br/>"+"d'erreur maximale ="+str(emax)+" "+ch
#####################"Méthode Newton" 
@app.route("/login_Newton",methods=['get', 'post'] )   # getvalue de la page /plott 
def getNewton():
    #nbrPoints=subject
    # listPt = name  
    #saisir fonction = subjetct1 
    #degP = subject2
    try:
        subject = request.form['subject']    # ces variables sont de type str 
        name = request.form['name']

        subject1 = request.form['subject1']
        subject2 = request.form['subject2']
        subject3 = request.form['subject3']
        
        #warning=""
        #######"fonctions "
        def saisie_facultatif(f) :
            warning=''
            """éviter les espaces"""
            """éviter les puissances négatives"""
            """les constantes pi et e ne sont pas définie"""
            """la variable que x ou X"""
            """-x n'est pas accepté --> 0-x"""
            
            if f == "" :
                return "",warning
            test = 1

            #nombre de parenthèse ouvrant égal nombre de parenthèse fermant 
            openn = []
            close = []
            for i in range ( len(f) ) :
                if f[i] == "(" :
                    openn.append(i)
                elif f[i] == ")" :
                    close.append(i)
            if len(openn) != len(close):
                warning+='il manque un parenthèse\n'
                
            #vérifier la forme de l'éxpression
            if '***' in f:
                warning+='vérifier les opérants * et **\n'
                
            i = 0
            while i < len(f)-1:
                if (f[i:i+3] =='log') :
                    #verification de la base de log
                    par = f[i:].find('(') + i
                    try :
                        float(f[i+3 : par])
                    except:
                        test = 0
                        #yield i
                        warning+='verifier la fonction log :: position '+str(i)+'\n'
                        break
                    #à se niveau log est suivi d'une base la forme est correcte exp log10
                    #verivication de l'expression entre les parenthéses de log exp log10(x**2)
                    if f[par + 1] in [')','+','-','*','/'] :
                        #exp log2() ou log2(-x) {la forme correcte log2(0-x)}
                        test = 0 
                        #yield i
                        warning+='verifier la fonction log :: position '+str(i)+'\n'
                        break
                    i += par + 1
                    #print(i)
                    continue
                if (f[i].isnumeric()) :
                    #n est numérique ce qui est vraie: 
                    # n) , n+ , n- , n* , n/ , ou suivie d'un autre chiffre n1 exp 21 , ou virgule 5.
                    # (n , +n , -n , *n , /n , ou précédé d'un autre chiffre 2n exp 22 , ou virgule .2
                    if (f[i + 1] not in [')','+','-','*','/','0', '1','2','3', '4', '5', '6', '7', '8', '9','.']) or (f[i - 1] not in ['(','+','-','*','/','0', '1','2','3', '4', '5', '6', '7', '8', '9','.']) :
                        test = 0 
                        #yield i
                        warning +='un opérant ou des parenthèses manquent :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i].lower() == 'x') :
                    #comme le cas numérique mais on peut avoir 2x ou x2
                    if (f[i + 1] not in [')','+','-','*','/']) or (f[i - 1] not in ['(','+','-','*','/']) :
                        test = 0 
                        #yield i
                        warning+='un opérant ou des parenthèses manquent :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if f[i] == ')' :
                    #les cas corrects: x) , chiffre) , ))
                    # )+ , )- , )* , )/ , )) 
                    if (f[i - 1] not in ['x', 'X', '0', '1','2','3', '4', '5', '6', '7', '8', '9', ')']) or (f[i + 1] not in [')','+','-','*','/']) :
                        test = 0 
                        #yield i 
                        warning+='vérifier les parenthèses :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i] == '*'):
                    # x* , chiffre* , )* , **
                    # les cas à éviter : *) , *+ , *- , */ 
                    if (f[i - 1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')','*']) or (f[i + 1] in [')','+','-','/']):
                        test = 0
                        #yield i 
                        warning+='vérifier les opérants :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i] in ['+','/','-' ] ):
                    # les cas possibles x+ , chiffre+ , )+ de même pour / et -
                    # les cas à éviter : +) , ++ , +- , +/ , +. de même pour / et -
                    if (f[i - 1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')']) or (f[i + 1] in [')','+','-','*','/','.']):
                        test = 0
                        #yield i 
                        warning+= 'vérifier les opérants :: position '+str(i)+'\n'
                        break
                    if (f[i] == '/') and (f[i+1] == '0') and f[i+2] != '.':
                        test = 0
                        #yield i 
                        warning+= 'diviseur de zéro :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                i += 1
            #verifier le dernier element de l'expression
            if f[len(f)-1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')']:
                test = 0
                warning+= "l'expression n'a pas terminé correctement \n"
            #if test == 1 :
            #l'expression verifie toutes les contraites
            return f,warning
                    
        def  calcul_simple (expression, x) :
            warning=""
        
            #expression en fonction de x (et/ou X) sans parenthése ne contient que des opérants parmi {+, -, /, *, **} 
            # exemple "x*5**2+X/2+1"
            expression = expression.replace('X','x')
            expression = expression.replace('x',str(x))

            l = []
            i=1
            while i< len(expression):
                if expression[i] in ['+','-','*','/']:
                    l.append(float(expression[:i]))
                    if expression[i+1] == "*":
                            l.append('**')
                            expression = expression[i+2 :]
                            i=0
                    else:
                        l.append(expression[i])
                        expression = expression[i+1 :]
                        i=0
                i += 1
            l.append(float(expression))
            try :
                #calcul des puissances
                i = 1
                while i < len(l) :
                    if l[i] == '**' :
                        y = l[: i-1] + [float(l[i-1]) ** float(l[i+1])] + l[i+2 : ]
                        l = y
                    else:
                        i += 1
                #calcul des * et /
                i = 1
                while i < len(l) :
                    if l[i] == '*' :
                        y = l[:i-1] + [float(l[i-1]) * float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    if l[i] == '/' :
                        y = l[:i-1] + [float(l[i-1]) / float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    i += 1
                #calcul des + et -
                i = 1
                while i < len(l):
                    if l[i] == '+' :
                        y = l[:i-1] + [float(l[i-1]) + float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    if l[i] == '-' :
                        y = l[:i-1] + [float(l[i-1]) - float(l[i+1])] + l[i+2 : ]
                        l = y
                        continue
                    i += 1
            except :
                testing=0
                warning += "erreur: paramètre multiple \nerreur: F doit être en fonction de x \n "

            return l[0]

        def calcul_avance (fonction, x):
            #fonction "tan" ou "e"
            
            import math as m
            if fonction == 'tan' : return round(m.tan(x),15)
            elif fonction == 'sin' : return round(m.sin(x),15)
            elif fonction == 'cos' : return round(m.cos(x),15)
            elif fonction == 'asin' : return round(m.asin(x),15)
            elif fonction == 'acos' : return round(m.acos(x),15)
            elif fonction == 'atan' : return round(m.atan(x),15)
            elif fonction == 'sinh' : return round(m.sinh(x),15)
            elif fonction == 'cosh' : return round(m.cosh(x),15)
            elif fonction == 'tanh' : return round(m.tanh(x),15)
            elif fonction == 'e' : return round(m.exp(x),15)
            elif fonction == 'ln' : return round(m.log(x),15)
            elif fonction == 'log' : return round(m.log10(x),15)
            elif fonction == 'E' : return round(m.floor(x),15)
            elif fonction == 'factorial' : return round(m.factorial(x),15)
            elif fonction[:3] == 'log' :
                    base = int(fonction[3:])
                    return round(m.log(x, base),15)
            else:
                return "fonction n'existe pas"
        def image_par_fonction(expression,x):
            #expression chaine de caractère "sin(x)+2*(exp(-2x)+5)"
            #y = F(x)
            i=0
            openn = []
            test = True
            while i < len(expression):
                if expression[i] == "(" :
                    openn.append(i)
                    i += 1
                elif expression[i] == ")" :
                    block = expression[openn[-1] +1: i]
                    ch2 = expression[i + 1 :]
                    val_block = calcul_simple (block, x)
                    if expression[openn[-1] - 1].isalpha() or  expression[openn[-1] - 1].isnumeric():
                        j = openn[-1] - 1
                        while expression[j] not in ["+", "-", "/", "*", "**", "("] and j>=0 : 
                            j -= 1
                        fonction = expression[j+1 : openn[-1]]
                        
                        val = calcul_avance(fonction, val_block)
                        ch1 = expression [: j+1]
                    else:
                        val = val_block
                        ch1 = expression[:openn[-1]]
            
                    if type(val) == type("A") :
                        test = False
                        break
                    expression = ch1 + str(val)
                    i = len(expression)
                    expression += ch2
                    openn.pop( len(openn)-1 )
                else: 
                    i += 1
            if test  :    
                return calcul_simple(expression, x)
            return val

        def image_par_interpolation(x,coef_poly_inter):
            # coef_poly_inter est une liste contenant les coefficients du polynome d'interpolation
            n = len(coef_poly_inter) - 1
            y = 0
            for ai in coef_poly_inter:
                y += ai * x**n
                n -= 1
            return y

        #x="(0;0.107981),(0.5;0.483941),(1;0.797885),(1.5;0.483941)"
        def diff_divisee(lx,ly):
            n=len(lx)
            m=np.zeros((n,n-1))
            m=np.c_[ly,m]
            
            for i in range(n):
                for j in range(1,i+1):
                    v=(m[i][j-1]-m[i-1][j-1])/(lx[i]-lx[i-j])
                    m[i][j]=v
            return [m[i][i] for i in range(n)]

        def poly_inter_newton(L,deg):
            x=[]
            y=[]
            n= len(L)
            for i in range(n):
                x.append(L[i][0])
                y.append(L[i][1])
            m=diff_divisee(x,y)
            coef_poly_new =[m[0]]+[0 for i in range(deg)]
            for i in range(n):
                x[i] *= 0-1
            for k in range(1,deg+1):
                lx=x[:k]
                l=[m[k]]
                for r in range(1,len(lx)+1):
                    ai = 0
                    comb = combinaisons(lx, r)
                    for c in comb:
                        prod=1
                        for j in range(r):
                            prod *= c[j]
                        ai += prod*m[k]
                    l.append(ai)
                for i in range(k,-1,-1):
                    coef_poly_new[k-i]+=l[i]
            
            coef_poly_new=list(reversed(coef_poly_new))
            return coef_poly_new

        def combinaisons(iterable, r):
            # combinaisons('ABCD', 2) --> AB AC AD BC BD CD
            # combinaisons(range(4), 3) --> 012 013 023 123
            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))
            yield tuple(pool[i] for i in indices)
            while True:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)

        def erreur (L,P,F):
            # renvoie max|F(x)-P(x)|
            #P : [an, an-1, .., a0]
            #F : chaine exemple "exp(x)+sin(x)"
            #L : liste des coordonnées des points
            #lx liste des xi
            if F == "":
                return ""
            lx = []
            for point in L:
                lx.append(point[0])

            emax=0
            posMax=None
            for i in range(len(lx)-1):
                vect = np.linspace(lx[i], lx[i+1], 10)
                
                for x in vect:
                    e = (image_par_fonction(F,x)-image_par_interpolation(x, P))
                    if abs(e) > abs(emax) :
                        emax = e
                        posMax=x
            return (emax,posMax)

        def points(ch):
            if name !="":
                l=ch.split(",")
                point=[]
                for e in l:
                    sep=e.index(";")
                    x=e[1:sep]
                    y=e[sep+1:-1]
                    a=float(x)
                    b=float(y)
                    point.append((a,b))
                return (point)
            else:
                return []

        def expression(P):
            #P liste des coefficient du polynome
            n=len(P)
            ch=""
            for i in range (n):
                if P[i]==0:
                    continue
                elif ch!='':
                    if P[i]>0:
                        ch+="+"
                if n-i-1 == 1 or n-i-1 == 0:
                    if P[i]==1:
                        ch+=" x "*(n-i-1)
                    else:
                        ch+=str(P[i])+" x "*(n-i-1)
                else:
                    if P[i]==1:
                        ch+=" x**"+str(n-1-i)+" "
                    else:
                        ch+=str(P[i])+" x**"+str(n-1-i)+" "
            return ch

        
        def approxErr(points,deg,x):
            lx=[]
            ly=[]
            n= len(points)
            for i in range(n):
                lx.append(points[i][0])
                ly.append(points[i][1])
            m=diff_divisee(lx,ly)
            prod=1
            fact=1
            for i in range(deg+1):
                prod*=(x-lx[i])
                fact*=i+1
            err=m[deg]*prod/fact
            return err
        L=points(name) 
        if len (L)!=int (subject):
            return render_template("about.html",subject=subject)    # here 

        m=saisie_facultatif(subject1)
        F=m[0]
        warning=m[1]

        #F=" (1/(0.5*(2*3.1415926)**0.5)*e((0-0.5)*(x-1)**2/0.5**2))"
        deg=int(subject2)
        P=poly_inter_newton(L,deg)  
        if warning !="":
            return render_template("about.html",F=F,warning=warning)
        else:
            T= erreur (L,P,F)
            if T != "":
                emax=T[0]
                posMaxErr=T[1]
            ######""#
            
            fig = Figure(figsize=(10,6))
            axis = fig.subplots()
            L.sort()
            lx = []
            ly = []
            for point in L:
                lx.append(point[0]) 
                ly.append(point[1])
            axis.plot(lx, ly,'x', label="Points")
            x = np.linspace(lx[0], lx[-1], 100)
            Py= []
            Fy= []

            if (P != None):
                for j in x:
                    Py.append(image_par_interpolation(j, P))
                axis.plot(x, Py, label="P(x)")
            if (F != ""):
                for j in x:
                    Fy.append(image_par_fonction(F,j))
                axis.plot(x, Fy, label="F(x)")
                if (posMaxErr != None):
                    a=image_par_fonction(F,posMaxErr)
                    b=image_par_interpolation(posMaxErr, P) #here
                    axis.plot([posMaxErr,posMaxErr],[min(b,a),max(b,a)],'r')
            if subject3!="":
                x=float(subject3)
                apEr=approxErr(L,deg,x)
                apimx=image_par_interpolation(x, P)
            else:
                x="x"
                apEr="NONE"
                apimx="NONE"
            # Save it to a temporary buffer.
            buf = BytesIO()
            fig.savefig(buf, format="png")
            # Embed the result in the html output.
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            ch1 =""
            ch2 =""
            ch3 =""
            expr=expression(P)
            ch1="La liste des points entrées ="+str(L)
            if T != "":
                ch3="Pour minimiser l'erreur ,veuillez ajouter un point à l'abscisse :"+str(posMaxErr)
                ch2="L'erreur maximale ="+str(emax)
        
            return render_template("resultatNewton.html",data=data, ch1=ch1,ch2=ch2,ch3=ch3,expr=expr,x=x,apEr=apEr,apimx=apimx)
    except:
        cons="Voici quelques conseils:"
        univar="*verifier que la fonction est univariable"
        predef="*vérifier les noms des fonction prédéfinies"
        cnx="*vérifier la connexion"
        return render_template("about.html",cons=cons,univar=univar,predef=predef,cnx=cnx)

############
#####################"Méthode Hermite" 
@app.route("/login_Hermite",methods=['get', 'post'] )   # getvalue de la page /plott 
def getHermite():
    #nbrPoints=subject
    # listPt = name  
    #saisir fonction = subjetct1 
    #degP = subject2
    try:
        subject = request.form['subject']    # ces variables sont de type str 
        name = request.form['name']

        subject1 = request.form['subject1']
        subject2 = request.form['subject2']
        subject3 = request.form['subject3']
        
        #warning=""
        #######"fonctions "
        def saisie_facultatif(f) :
            warning=''
            """éviter les espaces"""
            """éviter les puissances négatives"""
            """les constantes pi et e ne sont pas définie"""
            """la variable que x ou X"""
            """-x n'est pas accepté --> 0-x"""
            
            if f == "" :
                return "",warning
            #test = 1

            #nombre de parenthèse ouvrant égal nombre de parenthèse fermant 
            openn = []
            close = []
            for i in range ( len(f) ) :
                if f[i] == "(" :
                    openn.append(i)
                elif f[i] == ")" :
                    close.append(i)
            if len(openn) != len(close):
                warning+='il manque un parenthèse\n'
                
            #vérifier la forme de l'éxpression
            if '***' in f:
                warning+='vérifier les opérants * et **\n'
                
            i = 0
            while i < len(f)-1:
                if (f[i:i+3] =='log') :
                    #verification de la base de log
                    par = f[i:].find('(') + i
                    try :
                        float(f[i+3 : par])
                    except:
                        #test = 0
                        #yield i
                        warning+='verifier la fonction log :: position '+str(i)+'\n'
                        break
                    #à se niveau log est suivi d'une base la forme est correcte exp log10
                    #verivication de l'expression entre les parenthéses de log exp log10(x**2)
                    if f[par + 1] in [')','+','-','*','/'] :
                        #exp log2() ou log2(-x) {la forme correcte log2(0-x)}
                        #test = 0 
                        #yield i
                        warning+='verifier la fonction log :: position '+str(i)+'\n'
                        break
                    i += par + 1
                    #print(i)
                    continue
                if (f[i].isnumeric()) :
                    #n est numérique ce qui est vraie: 
                    # n) , n+ , n- , n* , n/ , ou suivie d'un autre chiffre n1 exp 21 , ou virgule 5.
                    # (n , +n , -n , *n , /n , ou précédé d'un autre chiffre 2n exp 22 , ou virgule .2
                    if (f[i + 1] not in [')','+','-','*','/','0', '1','2','3', '4', '5', '6', '7', '8', '9','.']) or (f[i - 1] not in ['(','+','-','*','/','0', '1','2','3', '4', '5', '6', '7', '8', '9','.']) :
                        #test = 0 
                        #yield i
                        warning +='un opérant ou des parenthèses manquent :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i].lower() == 'x') :
                    #comme le cas numérique mais on peut avoir 2x ou x2
                    if (f[i + 1] not in [')','+','-','*','/']) or (f[i - 1] not in ['(','+','-','*','/']) :
                        #test = 0 
                        #yield i
                        warning+='un opérant ou des parenthèses manquent :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if f[i] == ')' :
                    #les cas corrects: x) , chiffre) , ))
                    # )+ , )- , )* , )/ , )) 
                    if (f[i - 1] not in ['x', 'X', '0', '1','2','3', '4', '5', '6', '7', '8', '9', ')']) or (f[i + 1] not in [')','+','-','*','/']) :
                        #test = 0 
                        #yield i 
                        warning+='vérifier les parenthèses :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i] == '*'):
                    # x* , chiffre* , )* , **
                    # les cas à éviter : *) , *+ , *- , */ 
                    if (f[i - 1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')','*']) or (f[i + 1] in [')','+','-','/']):
                        #test = 0
                        #yield i 
                        warning+='vérifier les opérants :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                if (f[i] in ['+','/','-' ] ):
                    # les cas possibles x+ , chiffre+ , )+ de même pour / et -
                    # les cas à éviter : +) , ++ , +- , +/ , +. de même pour / et -
                    if (f[i - 1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')']) or (f[i + 1] in [')','+','-','*','/','.']):
                        #test = 0
                        #yield i 
                        warning+= 'vérifier les opérants :: position '+str(i)+'\n'
                        break
                    if (f[i] == '/') and (f[i+1] == '0') and f[i+2] != '.':
                        #test = 0
                        #yield i 
                        warning+= 'diviseur de zéro :: position '+str(i)+'\n'
                        break
                    i += 1
                    continue
                i += 1
            #verifier le dernier element de l'expression
            if f[len(f)-1] not in ['X','x','0', '1','2','3', '4', '5', '6', '7', '8', '9',')']:
                #test = 0
                warning+= "l'expression n'a pas terminé correctement \n"
            #if test == 1 :
            #l'expression verifie toutes les contraites
            return (f,warning)
                    
        def  calcul_simple (expression, x) :
            #expression en fonction de x (et/ou X) sans parenthése ne contient que des opérants parmi {+, -, /, *, **} 
            # exemple "x*5**2+X/2+1"
            expression = expression.replace('X','x')
            expression = expression.replace('x',str(x))

            l = []
            i=1
            while i< len(expression):
                if expression[i] in ['+','-','*','/']:
                    l.append(round(float(expression[:i]),15))
                    if expression[i+1] == "*":
                        l.append('**')
                        expression = expression[i+2 :]
                        i=0
                    else:
                        l.append(expression[i])
                        expression = expression[i+1 :]
                        i=0
                i += 1
            l.append(round(float(expression),15))
            try :
                #calcul des puissances
                i = 1
                while i < len(l) :
                    if l[i] == '**' :
                        y = l[: i-1] + [round(float(l[i-1]) ** float(l[i+1]),15)] + l[i+2 : ]
                        l = y
                    else:
                        i += 1
                #calcul des * et /
                i = 1
                while i < len(l) :
                    if l[i] == '*' :
                        y = l[:i-1] + [round(float(l[i-1]) * float(l[i+1]),15)] + l[i+2 : ]
                        l = y
                        continue
                    if l[i] == '/' :
                        y = l[:i-1] + [round(float(l[i-1]) / float(l[i+1]),15)] + l[i+2 : ]
                        l = y
                        continue
                    i += 1
                #calcul des + et -
                i = 1
                while i < len(l):
                    if l[i] == '+' :
                        y = l[:i-1] + [round(float(l[i-1]) + float(l[i+1]),15)] + l[i+2 : ]
                        l = y
                        continue
                    if l[i] == '-' :
                        y = l[:i-1] + [round(float(l[i-1]) - float(l[i+1]),15)] + l[i+2 : ]
                        l = y
                        continue
                    i += 1
            except :
                return "erreur: paramètre multiple \nerreur: F doit être en fonction de x "
            
            return l[0]

        def calcul_avance (fonction, x):
            #fonction "tan" ou "e"
            
            import math as m
            if fonction == 'tan' : return round(m.tan(x),15)
            elif fonction == 'sin' : return round(m.sin(x),15)
            elif fonction == 'cos' : return round(m.cos(x),15)
            elif fonction == 'asin' : return round(m.asin(x),15)
            elif fonction == 'acos' : return round(m.acos(x),15)
            elif fonction == 'atan' : return round(m.atan(x),15)
            elif fonction == 'sinh' : return round(m.sinh(x),15)
            elif fonction == 'cosh' : return round(m.cosh(x),15)
            elif fonction == 'tanh' : return round(m.tanh(x),15)
            elif fonction == 'e' : return round(m.exp(x),15)
            elif fonction == 'ln' : return round(m.log(x),15)
            elif fonction == 'log' : return round(m.log10(x),15)
            elif fonction == 'E' : return round(m.floor(x),15)
            elif fonction == 'factorial' : return round(m.factorial(x),15)
            elif fonction[:3] == 'log' :
                    base = int(fonction[3:])
                    return round(m.log(x, base),15)
            else:
                return "fonction n'existe pas"

        def image_par_fonction(expression,x):
            #expression chaine de caractère "sin(x)+2*(exp(-2x)+5)"
            #y = F(x)
            i=0
            openn = []
            test = True
            while i < len(expression):
                if expression[i] == "(" :
                    openn.append(i)
                    i += 1
                elif expression[i] == ")" :
                    block = expression[openn[-1] +1: i]
                    ch2 = expression[i + 1 :]
                    val_block = calcul_simple (block, x)
                    if expression[openn[-1] - 1].isalpha() or  expression[openn[-1] - 1].isnumeric():
                        j = openn[-1] - 1
                        while expression[j] not in ["+", "-", "/", "*", "**", "("] and j>=0 : 
                            j -= 1
                        fonction = expression[j+1 : openn[-1]]
                        
                        val = calcul_avance(fonction, val_block)
                        ch1 = expression [: j+1]
                    else:
                        val = val_block
                        ch1 = expression[:openn[-1]]
            
                    if type(val) == type("A") :
                        test = False
                        break
                    expression = ch1 + str(val)
                    i = len(expression)
                    expression += ch2
                    openn.pop( len(openn)-1 )
                else: 
                    i += 1
            if test  :    
                return calcul_simple(expression, x)
            return val

        def diff_divisee_h(lx,ly,dy):
            n=len(lx)
            m=np.zeros((n,n-1))
            m=np.c_[ly,m]
            for i in range(n):
                v=0
                if i%2!=0:
                    m[i][1]=dy[i//2]
                else:
                    v=(m[i][0]-m[i-1][0])/(lx[i]-lx[i-1])
                    m[i][1]=v
            for i in range(n):
                for j in range(2,i+1):
                    v=(m[i][j-1]-m[i-1][j-1])/(lx[i]-lx[i-j])
                    m[i][j]=v
            return [m[i][i] for i in range(n)]

        def poly_inter_hermite(L,deg):
            x=[]
            y=[]
            dy=[]
            n= len(L)
            for i in range(n):
                x.append(L[i][0])
                y.append(L[i][1])
                if i%2==0:
                    dy.append(L[i][2])
            m=diff_divisee_h(x,y,dy)
            coef_poly_her =[m[0]]+[0 for i in range(deg)]
            for i in range(n):
                x[i] *= 0-1
            for k in range(1,deg+1):
                lx=x[:k]
                l=[m[k]]
                for r in range(1,len(lx)+1):
                    ai = 0
                    comb = combinaisons(lx, r)
                    for c in comb:
                        prod=1
                        for j in range(r):
                            prod *= c[j]
                        ai += prod*m[k]
                    l.append(ai)
                for i in range(k,-1,-1):
                    coef_poly_her[k-i]+=l[i]
            
            coef_poly_her=list(reversed(coef_poly_her))
            return coef_poly_her

        def image_par_interpolation(x,coef_poly_inter):
            # coef_poly_inter est une liste contenant les coefficients du polynome d'interpolation
            n = len(coef_poly_inter) - 1
            y = 0
            for ai in coef_poly_inter:
                y += ai * x**n
                n -= 1
            return y

        #x="(0;0.107981),(0.5;0.483941),(1;0.797885),(1.5;0.483941)" 
        def combinaisons(iterable, r):
            # combinaisons('ABCD', 2) --> AB AC AD BC BD CD
            # combinaisons(range(4), 3) --> 012 013 023 123
            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))
            yield tuple(pool[i] for i in indices)
            while True:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)

        def erreur (L,P,F):
            # renvoie max|F(x)-P(x)|
            #P : [an, an-1, .., a0]
            #F : chaine exemple "exp(x)+sin(x)"
            #L : liste des coordonnées des points
            #lx liste des xi
            if F == "":
                return ""
            lx = []
            for i in range(len(L)):
                    lx.append(L[i][0])

            emax=0
            posMax=None
            for i in range(len(lx)-1):
                vect = np.linspace(lx[i], lx[i+1], 10)
                
                for x in vect:
                    e = (image_par_fonction(F,x)-image_par_interpolation(x, P) )
                    if abs(e) > abs(emax):
                        emax = e
                        posMax=x
            return (emax,posMax)

        def Points(ch):
            if name !="":
                l=ch.split(",")
                point=[]
                for e in l:
                    sep1=e.index(";")
                    x=e[1:sep1]
                    sep2=e.index(";",sep1+1)
                    y=e[sep1+1:sep2]
                    dy=e[sep2+1:len(e)-1]
                    a=float(x)
                    b=float(y)
                    c=float(dy)
                    point.append((a,b,c))
                return (point)
            else:
                return []

        def expression(P):
            #P liste des coefficient du polynome
            n=len(P)
            ch=""
            for i in range (n):
                if P[i]==0:
                    continue
                elif ch!='':
                    if P[i]>0:
                        ch+="+"
                if n-i-1 == 1 or n-i-1 == 0:
                    if P[i]==1:
                        ch+=" x "*(n-i-1)
                    else:
                        ch+=str(P[i])+" x "*(n-i-1)
                else:
                    if P[i]==1:
                        ch+=" x**"+str(n-1-i)+" "
                    else:
                        ch+=str(P[i])+" x**"+str(n-1-i)+" "
            return ch
    
        def approxErr(L,deg,x):
            lx=[]
            ly=[]
            dy=[]
            n= len(L)
            for i in range(n):
                lx.append(L[i][0])
                ly.append(L[i][1])
                if i%2==0:
                    dy.append(L[i][2])
            m=diff_divisee_h(lx,ly,dy)
            prod=1
            fact=1
            for i in range(deg+1):
                prod*=(x-lx[i])
                fact*=i+1
            err=m[deg]*prod/fact
            return err

        L=Points(name) 
        if len (L)!=int (subject):
            return render_template("about.html",subject=subject)    # here 

        m=saisie_facultatif(subject1)
        F=m[0]
        warning=m[1]
        p=[]
        for i in range(len(L)):
            p.append(L[i])
            p.append(L[i])
        #L=p.copy()
        #F=" (1/(0.5*(2*3.1415926)**0.5)*e((0-0.5)*(x-1)**2/0.5**2))"
        deg=int(subject2)
        P=poly_inter_hermite(p,deg)
        if warning !="":
            return render_template("about.html",F=F,warning=warning)
        else:
            T= erreur (L,P,F)
            if T != "":
                emax=T[0]
                posMaxErr=T[1]
            
            fig = Figure(figsize=(10,6))
            axis = fig.subplots()
            L.sort()
            lx = []
            ly = []
            for i in range(len(L)):
                    lx.append(L[i][0]) 
                    ly.append(L[i][1])
            axis.plot(lx, ly,'x', label="Points")
            
            x = np.linspace(lx[0], lx[-1], 100)
            Py= []
            Fy= []
            if (P != None):
                for j in x:
                    Py.append(image_par_interpolation(j, P))
                axis.plot(x, Py, label="P(x)")
            if (F != ""):
                for j in x:
                    Fy.append(image_par_fonction(F,j))
                axis.plot(x, Fy, label="F(x)")
                if (posMaxErr != None):
                    a=image_par_fonction(F,posMaxErr)
                    b=image_par_interpolation(posMaxErr, P) #here
                    axis.plot([posMaxErr,posMaxErr],[min(b,a),max(b,a)],'r')
            if subject3 != "":
                x=float(subject3)
                apEr=approxErr(p,deg,x)
                apimx=image_par_interpolation(x, P)
            else:
                x="NONE"
                apEr=""
                apimx=""


            # Save it to a temporary buffer.
            buf = BytesIO()
            fig.savefig(buf, format="png")
            # Embed the result in the html output.
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            ch1 =""
            ch2 =""
            ch3 =""
            expr=expression(P)
            ch1="La liste des points entrées ="+str(L)
            if T != "":
                ch3="Pour minimiser l'erreur ,veuillez ajouter un point à l'abscisse :"+str(posMaxErr)
                ch2="L'erreur maximale ="+str(emax)
            
            return render_template("resultatHermite.html",data=data, ch1=ch1,ch2=ch2,ch3=ch3,expr=expr,apEr=apEr,x=x,apimx=apimx)
    except:
        cons="Voici quelques conseils:"
        univar="*verifier que la fonction est univariable"
        predef="*vérifier les noms des fonction prédéfinies"
        cnx="*vérifier la connexion"
        return render_template("about.html",cons=cons,univar=univar,predef=predef,cnx=cnx)
############################debug 
