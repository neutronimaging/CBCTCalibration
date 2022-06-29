move(exp_shutter, 'open')
values = [27, 27, 27, 27.25, 27.25 ,27.25 , 27.5, 27.5, 27.5, 27.75, 27.75, 27.75, 28, 28, 28]
for i in values:
    maw(focus_midi, i)
    count()
move(exp_shutter, 'closed') 
