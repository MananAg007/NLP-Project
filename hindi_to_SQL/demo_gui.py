

# Import the required library
import tkinter
import tkinter.font as font
from tkinter import *
from tkinter import ttk
from demo import *
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate



# Create an instance of tkinter frame
win=Tk()
win.title("CS626 Hindi2SQL")

# Set the geometry
win.geometry("1600x900")


labelp=Label(win, text="Column Details", font=('Calibri 20'))
labelp.pack()


frame1 = LabelFrame(win, padx=20, pady=20)
frame1.pack()

week_rec_cols = [["Table 0"],["Week", "real", None], ["Date", "string", None], ["Opponent", "string", None], ["Result", "string", None], ["Record", "string", None], ["Game Site", "string", None], ["Attendance", "real", None]]
race_cols = [["Table 1"],["Season", "string", None], ["Series", "string", None], ["Team", "string", None], ["Races", "real", None], ["Wins", "real", None], ["Poles", "real", None], ["F/Laps", "real", None], ["Podiums", "real", None], ["Points", "real", None], ["Position", "string", None]]
party_cols = [["Table 2"],["District", "string", None], ["Incumbent", "string", None], ["Party", "string", None], ["First elected", "real", None], ["Result", "string", None], ["Candidates", "string", None]]
artist_cols = [["Table 3"],["Position", "real", None], ["Artist", "string", None], ["Song title", "string", None], ["Highest position", "real", None], ["Points", "real", None]]

c1 = [i[0] for i in week_rec_cols]
c2 = [i[0] for i in race_cols]
c3 = [i[0] for i in party_cols]
c4 = [i[0] for i in artist_cols]

lst = [c1, c2, c3, c4]


total_rows = len(lst)
total_columns = len(lst[0])



class Table:
    def __init__(self,root):
        # code for creating table
        for i in range(total_rows):
            for j in range(total_columns):
                if j==0:
                    self.e = Entry(root, width= 14, fg='white',
                               font=('Arial', 20, 'bold'))
                else:
                    self.e = Entry(root, width=12, fg='white',
                               font=('Arial',20))
                self.e.grid(row=i, column=j)
                try:
                    self.e.insert(END, lst[i][j])
                except:
                    pass

t= Table(frame1)


def get_input():
    inp0 = text.get(1.0, "end-1c")

    inp0 = transliterate(inp0, sanscript.ITRANS,sanscript.DEVANAGARI)
    inp1 = text2.get(1.0, "end-1c")

    op = project_demo(inp0, int(inp1))
    print(op)

    label2.config(text= op)

# Add a text widget

# Create a Label widget
label0=Label(win, text="??????????????? ?????????????????? ???????????? ????????????", font=('Calibri 20'))
label0.pack()
text=Text(win, width=100, height=8)
text.insert(END, "")
text.pack()
Font_tuple = ("Calibri MS", 20)
text.configure(font = Font_tuple)


# Create a Label widget
label1=Label(win, text="???????????? ??????????????????????", font=('Calibri 20'))
label1.pack()
text2=Text(win, width=80, height=3)
text2.insert(END, "")
text2.pack()
text2.configure(font = Font_tuple)

def transliterate_hindi():
    inp0 = text.get(1.0, "end-1c")

    op = transliterate(inp0, sanscript.ITRANS,sanscript.DEVANAGARI)

    label_hin.config(text= op)

buttonFont = font.Font(family='Calibri', size=20, weight='bold')
b=tkinter.Button(win, text="Transliterate", command=transliterate_hindi, font=buttonFont)
b.pack()


label_hin=Label(win, text="", font=('Calibri 20'), pady=20)
label_hin.pack()


b=tkinter.Button(win, text="Generate SQL", command=get_input, font=buttonFont)
b.pack()

# Create a Label widget
label2=Label(win, text="", font=('Calibri 20'))
label2.pack()

win.mainloop()
