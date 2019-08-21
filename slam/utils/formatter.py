from string import Formatter


class PartialFormatter(Formatter):

    def vformat(self, format_string, args, kwargs):

        tokens = []
        for lit, name, spec, conv in self.parse(format_string):
            lit = lit.replace('{', '{{').replace('}', '}}')
            if name is None:
                tokens.append(lit)
                continue

            conv = '!' + conv if conv else ''
            spec = ':' + spec if spec else ''

            fp = name.split('[')[0].split('.')[0]

            if not fp or fp.isdigit() or fp in kwargs:
                tokens.extend([lit, '{', name, conv, spec, '}'])
            else:
                tokens.extend([lit, '{{', name, conv, spec, '}}'])

        format_string = ''.join(tokens)
        return super().vformat(format_string, args, kwargs)
