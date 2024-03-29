###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

from textwrap import TextWrapper


def info_text_report(items: list) -> str:
    keys = [k for k, v in items]
    max_key_len = max(len(k) for k in keys)
    report = ""
    for k, v in items:
        wrapper = TextWrapper(
            width=80,
            initial_indent=k.ljust(max_key_len) + " : ",
            subsequent_indent=" " * max_key_len + " : ",
        )
        text = wrapper.fill(str(v))
        report += text + "\n"
    return report


def info_html_report(items: list) -> str:
    report = '<table class="iarray-info">'
    report += "<tbody>"
    for k, v in items:
        report += (
            "<tr>"
            '<th style="text-align: left">%s</th>'
            '<td style="text-align: left">%s</td>'
            "</tr>" % (k, v)
        )
    report += "</tbody>"
    report += "</table>"
    return report


class InfoReporter(object):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        items = self.obj.info_items
        return info_text_report(items)

    def _repr_html_(self):
        items = self.obj.info_items
        return info_html_report(items)
