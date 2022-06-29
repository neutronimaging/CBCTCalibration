maw(exp_shutter, 'open')
maw(he_shutter, 'open')
maw(fs_shutter, 'open')
#------------------
values = [1, 2,3,4,5,6,7,8,9,10,11,12]
for i in values:
    maw(sp2_tx, 4)
    timescan(40, delay=2)
    maw(sp2_tx, -5)
    timescan(40, delay=2)
#--------------------------
move(exp_shutter, 'closed')
move(he_shutter, 'closed')
move(fs_shutter, 'closed')