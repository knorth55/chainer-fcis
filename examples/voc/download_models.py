#!/usr/bin/env python

import fcis


def main():
    fcis.models.FCISResNet101.download('voc')
    fcis.models.FCISResNet101.download('voc_converted')


if __name__ == '__main__':
    main()
