import run_functions
import tkinter

window = tkinter.Tk()
window.title("GUI")
window.geometry('650x400')


label = tkinter.Label(window, text = "Which algorithm do you want?",font=("Arial Bold", 15)).pack()

tkinter.Button(window, text = "Kruskal MST and Algorithm MWM with User Input", command = run_functions.userTSP_Kruskal_MST_Algorithm_MWM).pack()
tkinter.Button(window, text = "Kruskal MST and Formulation MWM with User Input", command = run_functions.userTSP_Kruskal_MST_Formulation_MWM).pack()
tkinter.Button(window, text = "Prim MST and Algorithm MWM with User Input", command = run_functions.userTSP_Prim_MST_Algorithm_MWM).pack()
tkinter.Button(window, text = "Prim MST and Formulation MWM with User Input", command = run_functions.userTSP_Prim_MST_Formulation_MWM).pack()
tkinter.Button(window, text = "Formulation MST and Algorithm MWM with User Input", command = run_functions.userTSP_Formulation_MST_Algorithm_MWM).pack()
tkinter.Button(window, text = "Formulation MST and Formulation MWM with User Input", command = run_functions.userTSP_Formulation_MST_Formulation_MWM).pack()
tkinter.Button(window, text = "Kruskal MST and Algorithm MWM with TSPLIB Input", command = run_functions.TSPlib_Kruskal_MST_Algorithm_MWM).pack()
tkinter.Button(window, text = "Kruskal MST and Formulation MWM with TSPLIB Input ", command = run_functions.TSPlib_Kruskal_MST_Formulation_MWM).pack()
tkinter.Button(window, text = "Prim MST and Algorithm MWM with TSPLIB Input", command = run_functions.TSPlib_Prim_MST_Algorithm_MWM).pack()
tkinter.Button(window, text = "Prim MST and Formulation MWM with TSPLIB Input", command = run_functions.TSPlib_Prim_MST_Formulation_MWM).pack()
tkinter.Button(window, text = "Formulation MST and Algorithm MWM with TSPLIB Input", command = run_functions.TSPlib_Formulation_MST_Algorithm_MWM).pack()
tkinter.Button(window, text = "Formulation MST and Formulation MWM with TSPLIB Input", command = run_functions.TSPlib_Formulation_MST_Formulation_MWM).pack()


window.mainloop()


