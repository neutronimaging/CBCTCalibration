#!/usr/local/bin/python
import xlsxwriter
import json
import urllib

user = input("Enter User name: ")
project = input("Enter Project name: ")
urlname= 'https://api.github.com/repos/'+user+'/'+project+'/releases'
print(urlname)
workbook = xlsxwriter.Workbook(user+'-'+project+'.xlsx')
worksheet = workbook.add_worksheet(project)
worksheet_graph = workbook.add_worksheet('Status Chart')
bold = workbook.add_format({'bold': True})
chart = workbook.add_chart({'type': 'column'})
worksheet.write('A1', 'Releases', bold)
worksheet.write('B1', 'Download Count', bold)
row = 1
col = 0

try:
    response = urllib.request.urlopen(urlname)
except urllib.httperror as e:
	if e.code == 404:
		print("No repository available")
		worksheet.write(row, col, "No repository available")
	else:
		raise
else:
	# 200
	download_counts = json.loads(response.read())
	for item in download_counts:
		if 'tag_name' in item:
			worksheet.write(row, col, item['tag_name'])
			if item['assets']:
				for idx,asset in enumerate(item['assets']):
					worksheet.write(row, col + 1 +idx, asset['download_count'])
			row += 1
			continue
	chart.add_series({
		'categories': '='+project+'!$A$2:$A$'+str(row),
		'values':     '='+project+'!$B$2:$B$'+str(row),
		'line':       {'color': 'blue'},
		'data_labels': {'value': True}
	})
	chart.set_title ({'name': project+' Download Status'})
	chart.set_x_axis({'name': 'Released version'})
	chart.set_y_axis({'name': 'Download Count'})
	chart.set_size({'x_scale': 2.5, 'y_scale': 3})
	worksheet_graph.insert_chart('A1', chart)
	print("Exported result saved in exports folder")

finally:
	workbook.close()
