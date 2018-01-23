#!/usr/bin/env python

import fcis


def main():
    fcis.models.FCISResNet101.download('voc_trained')


if __name__ == '__main__':
    main()
