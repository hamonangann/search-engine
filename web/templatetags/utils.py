from django import template

register = template.Library()

@register.simple_tag
def add(num,arg):
    return int(num)+int(arg)