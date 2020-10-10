"""
Tiny script meant to generate the HTML tables offline (copy and paste them into the HTML).
"""
import re
import copy

s_str = """<tr>
    <td>##00</td>
    <td><audio controls="controls" src="data/obligation/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="data/variant/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="data/shoulder/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="data/grandmother/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="data/art/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="data/drain/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="data/bubble/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
</tr>"""

def mod_string(i, idx):
    # i: clip number from the dataset
    # idx: clip number on the website
    global s_str

    s = copy.deepcopy(s_str)
    for m in re.finditer('\.wav', s):
        s = s[:m.start() - 2] + str(i).zfill(2) + s[m.start():]
    for m in re.finditer('##', s):
        s = s[:m.start()] + str(idx) + s[m.start() + 4:]
    print(s)

if __name__ == '__main__':
    for idx, i in enumerate([8, 10, 12, 16, 17]):
        mod_string(i, idx + 1)
