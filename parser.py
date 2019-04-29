from ply import lex, yacc
import re


# @profile
def myparse(filepath, newfile):

    # ##############################@ LEXER #########################@

    tokens = (
        'STRING',
        'TAB',
        'NEWLINE',
    )


    def t_STRING(t):
        r'[-,.!?:/"#&éèàê\w\\\\[\]\(\)\s"\']+'
        print(t)
        return t

    def t_TAB(t):
        r'\t'
        print("    ")
        newfile.write("\t")
        return t

    def t_NEWLINE(t):
        r'\n'
        print("\n")
        newfile.write("\n")
        return t

    # ignore
    t_ignore = ' \r'

    # error handeling in lexer
    def t_error(t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)
        raise Exception("Illegal char")

    lexer = lex.lex()

    # ############################# PARSER ###########################@

    start = 'def'

    def p_file(p):
        '''def : text'''

    def p_text(p):
        '''text : line
                | line NEWLINE text'''

    def p_line(p):
        '''line : STRING
                | STRING TAB line'''

        if len(p) == 3:
            e = str(p[2])
        else:
            e = str(p[1])

        print(e)
        newfile.write(e)

    def p_error(p):
        print("Syntax error in input!")
        print(p)
        raise Exception("Syntax error")

    parser = yacc.yacc()

    with open(filepath, 'r') as myfile:
        parser.parse(myfile.read())

    print("\nparsing OK\n")

    myfile.close()


newfile = open("donnees/newdata.data", "w")

myparse("donnees/playlists.data", newfile)

newfile.close()
