import os
import re
import sys
import xml.etree.ElementTree as E
from optparse import OptionParser
from glob import glob
from zipfile import ZipFile

VERSION = "1.0"


class ABCoutput:
    def __init__(self, fnmext, pad, index, options):
        self.fnmext = fnmext
        self.pad = pad
        self.index = index
        self.options = options
        self.title = ""
        self.clefs = {}
        self.pageFmt = {}
        self.midiMap = []
        self.tabVceMap = {}
        self.koppen = {}
        self.volpan = 0
        self.tempo = ""
        self.tempo_units = (1, 4)
        self.mtr = ""
        self.key = "none"

    def getABC(self):
        # Implement the method to return the ABC notation as a string
        pass

    def writeall(self):
        # Implement the method to write the ABC notation to a file
        pass

    def mkHeader(self, gStfMap, partlist, midiMap, tabVceMap, koppen):
        # Implement the method to create the header for the ABC notation
        pass


class Measure:
    def __init__(self, part_index):
        self.part_index = part_index
        self.reset()

    def reset(self):
        self.ixm = 0
        self.divs = 0
        self.mtr = (4, 4)
        self.mdur = 0
        self.attr = ""
        self.lline = "|"
        self.rline = "|"
        self.lnum = ""
        self.ixp = 0


class Parser:
    def __init__(self, options):
        self.options = options
        self.gStfMap = []
        self.instMid = []
        self.drumInst = {}
        self.dirStk = {}
        self.repeat_str = {}
        self.msc = None
        self.isSib = False
        self.diafret = 0
        self.stafflines = 5
        self.ped = options.ped
        self.pedVce = None
        self.curStf = {}
        self.vce2stf = {}
        self.stfMap = {}
        self.clefMap = {}
        self.clefOct = {}
        self.curClef = {}
        self.stemDir = {}
        self.tabmap = {}
        self.hasStems = {}
        self.vceInst = {}
        self.midiMap = []
        self.tabVceMap = {}
        self.koppen = {}
        self.drumNotes = {}
        self.msc = None

    def parse(self, xmltxt):
        e = E.fromstring(xmltxt)
        self.mkTitle(e)
        self.doDefaults(e)
        partlist = self.doPartList(e)
        parts = e.findall('part')
        for ip, p in enumerate(parts):
            maten = p.findall('measure')
            self.locStaffMap(p, maten)
            self.msc.initVoices(newPart=1)
            aantalHerhaald = 0
            herhaalMaat = 0
            divisions = []
            self.msr = Measure(ip)
            while self.msr.ixm < len(maten):
                maat = maten[self.msr.ixm]
                herhaal, lbrk = 0, ''
                self.msr.reset()
                self.curalts = {}
                es = list(maat)
                for i, e in enumerate(es):
                    if e.tag == 'note':
                        self.doNote(e)
                    elif e.tag == 'attributes':
                        self.doAttr(e)
                    elif e.tag == 'direction':
                        self.doDirection(e, i, es)
                    elif e.tag == 'sound':
                        self.doDirection(maat, i, es)
                    elif e.tag == 'harmony':
                        self.doHarmony(e, i, es)
                    elif e.tag == 'barline':
                        herhaal = self.doBarline(e)
                    elif e.tag == 'backup':
                        dt = int(e.findtext('duration'))
                        if self.chkbug(dt, self.msr):
                            self.msc.incTime(-dt)
                    elif e.tag == 'forward':
                        dt = int(e.findtext('duration'))
                        if self.chkbug(dt, self.msr):
                            self.msc.incTime(dt)
                    elif e.tag == 'print':
                        lbrk = self.doPrint(e)
                self.msc.addBar(lbrk, self.msr)
                divisions.append(self.msr.divs)
                if herhaal == 1:
                    herhaalMaat = self.msr.ixm
                    self.msr.ixm += 1
                elif herhaal == 2:
                    if aantalHerhaald < 1:
                        self.msr.ixm = herhaalMaat
                        aantalHerhaald += 1
                    else:
                        aantalHerhaald = 0
                        self.msr.ixm += 1
                else:
                    self.msr.ixm += 1
            for rv in self.repeat_str.values():
                rv[0] = '[I:repeat %s %d]' % (rv[1], 1)
            vvmap = self.msc.outVoices(divisions, ip, self.isSib)
            self.addStaffMap(vvmap)
            self.addMidiMap(ip, vvmap)
            vvmapAll.update(vvmap)
        if vvmapAll:
            abcOut.mkHeader(self.gStfMap, partlist, self.midiMap, self.tabVceMap, self.koppen)
        else:
            self.info('nothing written, %s has no notes ...' % abcOut.fnmext)

    def mkTitle(self, e):
        title = e.findtext('work/work-title', '').strip()
        mvttl = e.findtext('movement-title', '').strip()
        composer, lyricist, credits = [], [], []
        for creator in e.findall('identification/creator'):
            if creator.text:
                if creator.get('type') == 'composer':
                    composer += [line.strip() for line in creator.text.split('\n')]
                elif creator.get('type') in ('lyricist', 'transcriber'):
                    lyricist += [line.strip() for line in creator.text.split('\n')]
        for rights in e.findall('identification/rights'):
            if rights.text:
                lyricist += [line.strip() for line in rights.text.split('\n')]
        for credit in e.findall('credit'):
            cs = ''.join(e.text or '' for e in credit.findall('credit-words'))
            credits += [re.sub(r'\s*[\r\n]\s*', ' ', cs)]
        credits = self.filterCredits(credits)
        if title:
            title = 'T:%s\n' % title.replace('\n', '\nT:')
        if mvttl:
            title += 'T:%s\n' % mvttl.replace('\n', '\nT:')
        if credits:
            title += '\n'.join(['T:%s' % c for c in credits]) + '\n'
        if composer:
            title += '\n'.join(['C:%s' % c for c in composer]) + '\n'
        if lyricist:
            title += '\n'.join(['Z:%s' % c for c in lyricist]) + '\n'
        if title:
            abcOut.title = title[:-1]
        self.isSib = 'Sibelius' in (e.findtext('identification/encoding/software') or '')
        if self.isSib:
            self.info('Sibelius MusicXMl is unreliable')

    def filterCredits(self, credits):
        # Implement the method to filter credits based on the filter level
        pass

    def doDefaults(self, e):
        if not self.options.doPageFmt:
            return
        d = e.find('defaults')
        if d is None:
            return
        mils = d.findtext('scaling/millimeters')
        tenths = d.findtext('scaling/tenths')
        if not mils or not tenths:
            return
        xmlScale = float(mils) / float(tenths) / 10
        space = 10 * xmlScale
        abcScale = space / 0.2117
        abcOut.pageFmt['scale'] = abcScale
        eks = 2 * ['page-layout/'] + 4 * ['page-layout/page-margins/']
        eks = [
            a + b
            for a, b in zip(
                eks,
                'page-height,page-width,left-margin,right-margin,top-margin,bottom-margin'.split(
                    ','
                ),
            )
        ]
        for i in range(6):
            v = d.findtext(eks[i])
            k = abcOut.pagekeys[i + 1]
            if not abcOut.pageFmt[k] and v:
                try:
                    abcOut.pageFmt[k] = float(v) * xmlScale
                except:
                    self.info('illegal value %s for xml element %s', (v, eks[i]))
                    continue

    def doPartList(self, e):
        for sp in e.findall('part-list/score-part'):
            midi = {}
            for m in sp.findall('midi-instrument'):
                x = [
                    m.findtext(p, self.options.midDflt[i])
                    for i, p in enumerate(['midi-channel', 'midi-program', 'volume', 'pan'])
                ]
                pan = float(x[3])
                if pan >= -90 and pan <= 90:
                    pan = (float(x[3]) + 90) / 180 * 127
                midi[m.get('id')] = [
                    int(x[0]),
                    int(x[1]),
                    float(x[2]) * 1.27,
                    pan,
                ]
                up = m.findtext('midi-unpitched')
                if up:
                    self.drumInst[m.get('id')] = int(up) - 1
            self.instMid.append(midi)
        ps = e.find('part-list')
        xs = self.getPartlist(ps)
        partlist, _ = self.parseParts(xs, {}, [])
        return partlist

    def locStaffMap(self, part, maten):
        vmap = {}
        self.vceInst = {}
        self.msc.vnums = {}
        self.hasStems = {}
        self.stfMap, self.clefMap = {}, {}
        ns = part.findall('measure/note')
        for n in ns:
            v = int(n.findtext('voice', '1'))
            if self.isSib:
                v += 100 * int(n.findtext('staff', '1'))
            self.msc.vnums[v] = 1
            sn = int(n.findtext('staff', '1'))
            self.stfMap[sn] = []
            if v not in vmap:
                vmap[v] = {sn: 1}
            else:
                d = vmap[v]
                d[sn] = d.get(sn, 0) + 1
            x = n.find('instrument')
            if x is not None:
                self.vceInst[v] = x.get('id')
            x, noRest = n.findtext('stem'), n.find('rest') is None
            if noRest and (not x or x != 'none'):
                self.hasStems[v] = 1
        vks = list(vmap.keys())
        if self.options.jscript or self.isSib:
            vks.sort()
        for v in vks:
            xs = [(n, sn) for sn, n in vmap[v].items()]
            xs.sort()
            stf = xs[-1][1]
            self.stfMap[stf].append(v)
            self.vce2stf[v] = stf
            self.curStf[v] = stf

    def addStaffMap(self, vvmap):
        part = []
        for stf, voices in sorted(self.stfMap.items()):
            locmap = [vvmap[iv] for iv in voices if iv in vvmap]
            nostem = [(iv not in self.hasStems) for iv in voices if iv in vvmap]
            if locmap:
                part.append(locmap)
                clef = self.clefMap.get(stf, 'treble')
                for i, iv in enumerate(locmap):
                    clef_attr = ''
                    if clef.startswith('tab'):
                        if nostem[i] and 'nostems' not in clef:
                            clef_attr = ' nostems'
                        if self.diafret and 'diafret' not in clef:
                            clef_attr += ' diafret'
                    abcOut.clefs[iv] = clef + clef_attr
        self.gStfMap.append(part)

    def addMidiMap(self, ip, vvmap):
        instr = self.instMid[ip]
        if instr.values():
            defInstr = list(instr.values())[0]
        else:
            defInstr = self.options.midDflt
        xs = []
        for v, vabc in vvmap.items():
            ks = sorted(self.drumNotes.items())
            ds = [(nt, step, midi, head) for (vd, nt), (step, midi, head) in ks if v == vd]
            id = self.vceInst.get(v, '')
            if id in instr:
                xs.append((vabc, instr[id] + ds))
            else:
                xs.append((vabc, defInstr + ds))
        xs.sort()
        self.midiMap.extend([midi for v, midi in xs])
        snaarmap = ['E', 'G', 'B', 'd', 'f', 'a', "c'", "e'"]
        diamap = '0,1-,1,1+,2,3,3,4,4,5,6,6+,7,8-,8,8+,9,10,10,11,11,12,13,13+,14'.split(',')
        for k in sorted(self.tabmap.keys()):
            v, noot = k
            snaar, fret = self.tabmap[k]
            if self.diafret:
                fret = diamap[int(fret)]
            vabc = vvmap[v]
            snaar = self.stafflines - int(snaar)
            xs = self.tabVceMap.get(vabc, [])
            xs.append(
                '%%%%map tab%d %s print=%s heads=kop%s\n' % (vabc, noot, snaarmap[snaar], fret)
            )
            self.tabVceMap[vabc] = xs
            self.koppen[fret] = 1


def vertaal(xmltxt, **options_parm):
    class Options:
        def __init__(self, **kwargs):
            self.u = 0
            self.b = 0
            self.n = 0
            self.c = 0
            self.v = 0
            self.d = 0
            self.m = 0
            self.x = 0
            self.t = 0
            self.stm = 0
            self.mnum = -1
            self.p = 'f'
            self.s = 0
            self.j = 0
            self.v1 = 0
            self.ped = 0
            for key, value in kwargs.items():
                setattr(self, key, value)

    options = Options(**options_parm)
    options.p = options.p.split(',') if options.p else []
    abcOut = ABCoutput('', '', 0, options)
    psr = Parser(options)
    try:
        psr.parse(xmltxt)
        abc_str = abcOut.getABC()
        info('%s written with %d voices' % (abcOut.fnmext, len(abcOut.clefs)), warn=0)
    except Exception as e:
        etype, value, traceback = sys.exc_info()
        info('** %s occurred: %s' % (etype, value), 0)
        abc_str = ''
    return abc_str, ''.join(info_list)


def main():
    ustr = '%prog [-h] [-u] [-m] [-c C] [-d D] [-n CPL] [-b BPL] [-o DIR] [-v V]\n'
    ustr += '[-x] [-p PFMT] [-t] [-s] [-i] [--v1] [--noped] [--stems] <file1> [<file2> ...]'
    parser = OptionParser(usage=ustr, version=str(VERSION))
    parser.add_option("-u", action="store_true", help="unfold simple repeats")
    parser.add_option(
        "-m", action="store", help="0 -> no %%MIDI, 1 -> minimal %%MIDI, 2-> all %%MIDI", default=0
    )
    parser.add_option(
        "-c", action="store", type="int", help="set credit text filter to C", default=0, metavar='C'
    )
    parser.add_option("-d", action="store", type="int", help="set L:1/D", default=0, metavar='D')
    parser.add_option(
        "-n",
        action="store",
        type="int",
        help="CPL: max number of characters per line (default 100)",
        default=0,
        metavar='CPL',
    )
    parser.add_option(
        "-b",
        action="store",
        type="int",
        help="BPL: max number of bars per line",
        default=0,
        metavar='BPL',
    )
    parser.add_option(
        "-o", action="store", help="store abc files in DIR", default='', metavar='DIR'
    )
    parser.add_option(
        "-v",
        action="store",
        type="int",
        help="set volta typesetting behaviour to V",
        default=0,
        metavar='V',
    )
    parser.add_option("-x", action="store_true", help="output no line breaks")
    parser.add_option(
        "-p",
        action="store",
        help="pageformat PFMT (cm) = scale, pageheight, pagewidth, leftmargin, rightmargin, topmargin, botmargin",
        default='',
        metavar='PFMT',
    )
    parser.add_option(
        "-j", action="store_true", help="switch for compatibility with javascript version"
    )
    parser.add_option(
        "-t",
        action="store_true",
        help="translate perc- and tab-staff to ABC code with %%map, %%voicemap",
    )
    parser.add_option(
        "-s", action="store_true", help="shift node heads 3 units left in a tab staff"
    )
    parser.add_option(
        "--v1", action="store_true", help="start-stop directions allways to first voice of staff"
    )
    parser.add_option("--noped", action="store_true", help="skip all pedal directions")
    parser.add_option("--stems", action="store_true", help="translate stem directions")
    options, args = parser.parse_args()
    if not args:
        parser.print_help()
        sys.exit(1)
    for fnm in args:
        if os.path.isdir(fnm):
            for fnm in glob(os.path.join(fnm, '*.musicxml')):
                with open(fnm, 'r') as f:
                    xmltxt = f.read()
                abc_str, info_list = vertaal(xmltxt, **vars(options))
                if abc_str:
                    with open(fnm.replace('.musicxml', '.abc'), 'w') as f:
                        f.write(abc_str)
                for line in info_list:
                    print(line)
        else:
            with open(fnm, 'r') as f:
                xmltxt = f.read()
            abc_str, info_list = vertaal(xmltxt, **vars(options))
            if abc_str:
                with open(fnm.replace('.musicxml', '.abc'), 'w') as f:
                    f.write(abc_str)
            for line in info_list:
                print(line)


if __name__ == '__main__':
    main()
