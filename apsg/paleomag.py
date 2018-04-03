# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from datetime import datetime
from .core import Vec3, Fol, Lin, Group
from .helpers import sind, cosd

__all__ = ['Core']


class Core(object):
    """``Core`` store palemomagnetic analysis data

    Keyword Args:
      info:
      name:
      filename:
      alpha:
      beta:
      bedding:
      volume:
      date:
      steps:
      a95:
      vectors:

    Returns:
      ``Core`` object instance

    """
    def __init__(self, **kwargs):
        self.info = kwargs.get('info', 'Default')
        self.name = kwargs.get('name', 'Default')
        self.filename = kwargs.get('filename', None)
        self.alpha = kwargs.get('alpha', 0)
        self.beta = kwargs.get('beta', 0)
        self.bedding = kwargs.get('bedding', Fol(0, 0))
        self.volume = kwargs.get('volume', 'Default')
        self.date = kwargs.get('date', datetime.now())
        self.steps = kwargs.get('steps', [])
        self.a95 = kwargs.get('a95', [])
        self.comments = kwargs.get('comments', [])
        self._vectors = kwargs.get('vectors', [])

    def __repr__(self):
        return 'Core:' + str(self.name)

    @classmethod
    def from_pmd(cls, filename):
        """Return ``Core`` instance generated from PMD file.

        Args:
          filename: PMD file

        Example:
          >>> d = Core.from_pmd('K509A2-1.PMD')

        """
        with open(filename, encoding='latin1') as f:
            d = f.read().splitlines()
        data = {}
        fields = {'Xc': slice(5, 14), 'Yc': slice(15, 24), 'Zc': slice(25, 34),
                  'a95': slice(69, 73)}
        data['info'] = d[0].strip()
        vline = d[1].strip()
        data['filename'] = filename
        data['name'] = vline[:10].strip()
        data['alpha'] = float(vline[10:20].strip().split('=')[1])
        data['beta'] = float(vline[20:30].strip().split('=')[1])
        strike = float(vline[40:50].strip().split('=')[1])
        dip = float(vline[40:50].strip().split('=')[1])
        data['bedding'] = Fol(strike + 90, dip)
        data['volume'] = float(vline[50:63].strip().split('=')[1].strip('m3'))
        data['date'] = datetime.strptime(vline[63:], '%m-%d-%Y %H:%M')
        data['steps'] = [ln[:4].strip() for ln in d[3:-1]]
        data['comments'] = [ln[73:].strip() for ln in d[3:-1]]
        data['a95'] = [float(ln[fields['a95']].strip()) for ln in d[3:-1]]
        data['vectors'] = []
        for ln in d[3:-1]:
            x = float(ln[fields['Xc']].strip())
            y = float(ln[fields['Yc']].strip())
            z = float(ln[fields['Zc']].strip())
            data['vectors'].append(Vec3((x, y, z)))
        return cls(**data)

    def write_pmd(self, filename=None):
        """Save ``Core`` instance to PMD file.

        Args:
          filename: PMD file

        Example:
          >>> d.write_pmd(filename='K509A2-1.PMD')

        """
        import os
        if filename is None:
            filename = sefl.filename
        ff = os.path.splitext(os.path.basename(filename))[0][:8]
        dt = self.date.strftime("%m-%d-%Y %H:%M")
        infoln = '{:<8}  α={:<7.1f} ß={:< 7.1f} s={:<7.1f} d={:< 7.1f} v={:7.1E}m3  {}'
        ln0 = infoln.format(ff, self.alpha, self.beta, *self.bedding.rhr, self.volume, dt)
        headln = 'STEP  Xc [Am²]  Yc [Am²]  Zc [Am²]  MAG[A/m]   Dg    Ig    Ds    Is  a95 '
        tb = []
        for ix, step in enumerate(self.steps):
            ln = '{:<4} {: 9.2E} {: 9.2E} {: 9.2E} {: 9.2E} {: >5.1f} {: >5.1f} {: >5.1f} {: >5.1f} {: >4.1f} {}'.format(step, *self.V[ix], self.MAG[ix], *self.geo[ix].dd, *self.geo[ix].dd, self.a95[ix], self.comments[ix])
            tb.append(ln)
        with open(filename, 'w') as pmdfile:
            print(self.info, file=pmdfile)
            print(ln0, file=pmdfile)
            print(headln, file=pmdfile)
            for ln in tb:
                print(ln, file=pmdfile)
            print(bytearray.fromhex("1a").decode(), file=pmdfile)

    @property
    def MAG(self):
        return [abs(v) / self.volume for v in self._vectors]

    @property
    def V(self):
        return Group(self._vectors, name=self.name)

    @property
    def geo(self):
        return Group(self._vectors, name=self.name).rotate(Lin(0, 90), self.alpha).rotate(Lin(self.alpha + 90, 0), self.beta)