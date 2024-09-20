import sqlite3

def ver_datos():
    conn = sqlite3.connect('../data/movimientos.db')
    c = conn.cursor()
    
    c.execute("SELECT * FROM movimientos")
    filas = c.fetchall()
    
    for fila in filas:
        print(fila)

    conn.close()

if __name__ == '__main__':
    ver_datos()
