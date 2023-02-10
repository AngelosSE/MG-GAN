import re

with open('environment.yml','r') as f1, open('environment_new.yml','w') as f2:
	contents = ''.join(f1.readlines())
	p = re.compile(r"(?<=[^=])=[^=^\s]+(?=[\s])")
	contents = re.sub(p,'',contents)
	f2.writelines(contents)

# Method for simplifying the versioning number: '
# re.sub(r'([\d]+)\.([\d]+)\.([\d]+)',r'\g<1>.\g<2>',contents)

# More intuitive way to perform the above
# re.sub(r'=([\S]+)=[\S]+','=\g<1>',contents))
	
#patrick hedström direktnummer: 013378008