from flask import Flask, render_template, request, jsonify
import sympy as sp
import numpy as np
import io
import base64

# Configuración crítica para evitar errores de hilos (RuntimeError)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

class LaplaceSolver:
    def __init__(self):
        self.t = sp.symbols('t', real=True)
        self.s = sp.symbols('s', complex=True)

    def solve_dynamic(self, data):
        t, s = self.t, self.s
        order = int(data.get('order', 1))
        
        # EL CAMBIO: Usamos sp.Rational para mantener fracciones exactas
        # Esto convierte el "1" o "0.5" del usuario en una fracción real de SymPy
        a = sp.Rational(data.get('coeff_a', 1) or 1)
        b = sp.Rational(data.get('coeff_b', 0) or 0)
        c = sp.Rational(data.get('coeff_c', 0) or 0)
        y0 = sp.Rational(data.get('y0', 0) or 0)
        yp0 = sp.Rational(data.get('yp0', 0) or 0)
        
        local_dict = {'t': t, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'pi': sp.pi}
        
        ft_raw = str(data.get('ft', '0')).strip()
        # Convertimos la entrada f(t) también a forma racional si es posible
        f_t = sp.parse_expr(ft_raw, local_dict=local_dict)
        
        F_s = sp.laplace_transform(f_t, t, s, nocds=True)
        if isinstance(F_s, tuple): F_s = F_s[0]

        Y = sp.symbols('Y')
        if order == 1:
            eq_s = sp.Eq(b * (s * Y - y0) + c * Y, F_s)
        else:
            eq_s = sp.Eq(a*(s**2*Y - s*y0 - yp0) + b*(s*Y - y0) + c*Y, F_s)

        Y_s_sol = sp.solve(eq_s, Y)[0]
        y_t_sol = sp.inverse_laplace_transform(Y_s_sol, s, t).doit()
        
        # Usamos simplify para que junte los términos en una sola fracción elegante
        return sp.simplify(Y_s_sol), sp.simplify(y_t_sol)

solver = LaplaceSolver()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        
        # 1. Resolver matemáticamente
        Y_s, y_t = solver.solve_dynamic(data)
        
        # 2. Construir la vista previa de la ecuación planteada
        order = int(data.get('order', 1))
        a, b, c = data.get('coeff_a'), data.get('coeff_b'), data.get('coeff_c')
        ft = data.get('ft')
        
        if order == 2:
            eq_text = f"{a}y'' + {b}y' + {c}y = {ft}"
        else:
            eq_text = f"{b}y' + {c}y = {ft}"

        # 3. Limpiar Heaviside para que la fórmula se vea más estética
        # Solo lo quitamos de la vista de texto, no del cálculo de la gráfica
        y_t_view = y_t.replace(sp.Heaviside(solver.t), 1) if y_t.has(sp.Heaviside) else y_t
        
        # 4. Generar la gráfica en formato base64
        plot_b64 = generate_plot(y_t)
        
        return jsonify({
            'success': True,
            'result': {
                'eq_inicial': f"\\text{{Ecuación: }} {eq_text}",
                'Y_s': f"Y(s) = {sp.latex(sp.simplify(Y_s))}",
                'y_t': f"y(t) = {sp.latex(sp.simplify(y_t_view))}"
            },
            'plot': plot_b64
        })
    except Exception as e:
        print(f"Error detectado: {e}")
        return jsonify({'success': False, 'error': str(e)})

def generate_plot(expr):
    t_sym = sp.symbols('t')
    # Convertimos la expresión de SymPy a una función numérica de NumPy
    f_num = sp.lambdify(t_sym, expr, modules=['numpy'])
    t_vals = np.linspace(0, 10, 500)
    
    # Manejo de errores para funciones constantes o resultados inválidos
    try:
        y_vals = np.vectorize(f_num)(t_vals)
    except:
        y_vals = np.zeros_like(t_vals)
    
    # Configuración visual de la gráfica (Fondo transparente para Modo Oscuro)
    plt.figure(figsize=(8, 4.5), facecolor='none')
    ax = plt.gca()
    ax.set_facecolor('none')
    
    # Colores de los ejes adaptables
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#888')
    
    plt.plot(t_vals, y_vals, color='#9d65ff', linewidth=2.5, label='y(t)')
    plt.fill_between(t_vals, y_vals, color='#9d65ff', alpha=0.15)
    plt.grid(True, linestyle='--', alpha=0.2)
    
    # Guardar en buffer de memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)