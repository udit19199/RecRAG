Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=5a3d524195c25268 ===
Gold sentences and facts:
Sentence 0: NCAA Men 's National Collegiate Volleyball Championship Sport College indoor volleyball Founded 1970 No. of teams 6 ( as of 2014 ) Country United States Most recent champion ( s ) Ohio State 2 ( 3 ) TV partner ( s ) ESPNU Official website NCAA.com
  Fact 0.0: The NCAA Men's National Collegiate Volleyball Championship is a college indoor volleyball event.
  Fact 0.1: The NCAA Men's National Collegiate Volleyball Championship was founded in 1970.
  Fact 0.2: The NCAA Men's National Collegiate Volleyball Championship had 6 teams as of 2014.
  Fact 0.3: The NCAA Men's National Collegiate Volleyball Championship is held in the United States.
  Fact 0.4: Ohio State is listed as the most recent champion, with a count of 2 (3).
  Fact 0.5: The TV partner of the NCAA Men's National Collegiate Volleyball Championship is ESPNU.
  Fact 0.6: The official website of the NCAA Men's National Collegiate Volleyball Championship is NCAA.com.
Sentence 1: NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2 -- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State ( 21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara , California ( UCSB ) Robertson Gymnasium UCLA ( 30 -- 5 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 1 Springfield Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 27 -- 8 ) 3 -- 1 UC Santa Barbara Ohio State 2 -- 0 Yale 1976 Details Muncie , Indiana ( Ball State ) Irving Gymnasium UCLA ( 15 -- 2 ) 3 -- 0 Pepperdine Ohio State 2 -- 0 Springfield 1977 Details Los Angeles ( UCLA ) Pauley Pavilion USC ( 18 -- 1 ) 3 -- 1 Ohio State Pepperdine 2 -- 0 Rutgers -- Newark 1978 Details Columbus , Ohio ( Ohio State ) St. John Arena Pepperdine ( 21 -- 4 ) 3 -- 2 UCLA Ohio State 2 -- 0 Rutgers -- Newark 1979 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 30 -- 0 ) 3 -- 1 USC Rutgers -- Newark 3 -- 2 Ball State 1980 Details Muncie , Indiana ( Ball State ) Irving Gymnasium USC ( 22 -- 6 ) 3 -- 1 UCLA Ohio State 3 -- 0 Rutgers -- Newark 1981 Details Santa Barbara , California ( UCSB ) UCSB Events Center UCLA ( 32 -- 3 ) 3 -- 2 USC Penn State 3 -- 1 Ohio State 1982 Details University Park , Pennsylvania ( Penn State ) Rec Hall UCLA ( 29 -- 0 ) 3 -- 0 Penn State USC 2 -- 1 Ohio State Details Columbus , Ohio ( Ohio State ) St. John Arena UCLA ( 27 -- 4 ) 3 -- 0 Pepperdine Ohio State 3 -- 1 Penn State 1984 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 38 -- 0 ) 3 -- 1 Pepperdine George Mason 3 -- 0 Ball State 1985 Details Pepperdine ( 25 -- 2 ) 3 -- 2 USC George Mason 3 -- 0 Ball State 1986 Details University Park , Pennsylvania ( Penn State ) Rec Hall Pepperdine ( 22 -- 7 ) 3 -- 2 USC Penn State 3 -- 0 Ohio State Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 28 -- 3 ) 3 -- 0 USC Penn State 3 -- 0 Ohio State 1988 Details Fort Wayne , Indiana ( IPFW ) ACWMC USC ( 34 -- 4 ) 3 -- 2 UC Santa Barbara Ball State 3 -- 1 George Mason 1989 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 29 -- 5 ) 3 -- 1 Stanford Penn State 3 -- 0 Ball State 1990 Details Fairfax , Virginia ( George Mason ) Patriot Center USC ( 26 -- 7 ) 3 -- 1 Long Beach State Ball State 3 -- 1 Rutgers -- Newark 1991 Details Honolulu , HI ( Hawaii ) Neal S. Blaisdell Center Long Beach State ( 31 -- 4 ) 3 -- 1 USC IPFW 3 -- 1 Penn State 1992 Details Muncie , Indiana ( Ball State ) John E. Worthen Arena Pepperdine ( 24 -- 4 ) 3 -- 0 Stanford Penn State 3 -- 0 IPFW 1993 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 3 ) 3 -- 0 Cal State Northridge Penn State 3 -- 2 Ohio State 1994 Details Fort Wayne , Indiana ( IPFW ) ACWMC Penn State ( 26 -- 3 ) 3 -- 2 UCLA Ball State 3 -- 0 IPFW 1995 Details Springfield , Massachusetts ( Springfield ) Springfield Civic Center UCLA ( 31 -- 1 ) 3 -- 0 Penn State Ball State 3 -- 1 Hawaii Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 26 -- 5 ) 3 -- 2 Hawaii Lewis and Penn State 1997 Details Columbus , Ohio ( Ohio State ) St. John Arena Stanford ( 27 -- 4 ) 3 -- 2 UCLA Ball State and Penn State 1998 Details Honolulu , HI ( Hawaii ) Stan Sheriff Center UCLA ( 28 -- 4 ) 3 -- 0 Pepperdine Lewis and Princeton 1999 Details Los Angeles ( UCLA ) Pauley Pavilion BYU ( 30 -- 1 ) 3 -- 0 Long Beach State IPFW and Penn State 2000 Details Fort Wayne , Indiana ( IPFW ) ACWMC UCLA ( 29 -- 5 ) 3 -- 0 Ohio State Penn State and Pepperdine 2001 Details Long Beach , California ( Long Beach State ) The Pyramid BYU ( 23 -- 4 ) 3 -- 0 UCLA Ohio State and Penn State 2002 Details University Park , Pennsylvania ( Penn State ) Rec Hall Hawaii ( 24 -- 8 ) † 3 -- 1 Pepperdine Ball State and Penn State 2003 Details Long Beach , California ( Long Beach State ) The Pyramid Lewis ( 29 -- 6 ) † 3 -- 2 BYU Penn State and Pepperdine Details Honolulu , HI ( Hawaii ) Stan Sheriff Center BYU ( 29 -- 4 ) 3 -- 2 Long Beach State Lewis and Penn State 2005 Details Los Angeles ( UCLA ) Pauley Pavilion Pepperdine ( 25 -- 2 ) 3 -- 2 UCLA Ohio State and Penn State 2006 Details University Park , Pennsylvania ( Penn State ) Rec Hall UCLA ( 26 -- 12 ) 3 -- 0 Penn State UC Irvine and IPFW 2007 Details Columbus , Ohio ( Ohio State ) St. John Arena UC Irvine ( 29 -- 5 ) 3 -- 1 IPFW Penn State and Pepperdine 2008 Details Irvine , California ( UC Irvine ) Bren Events Center Penn State ( 30 -- 1 ) 3 -- 1 Pepperdine Long Beach State and Ohio State 2009 Details Provo , Utah ( BYU ) Smith Fieldhouse UC Irvine ( 27 -- 5 ) 3 -- 2 USC ( 21 -- 11 ) Ohio State and Penn State Details Stanford , California ( Stanford ) Maples Pavilion Stanford ( 24 -- 6 ) 3 -- 0 Penn State ( 24 -- 8 ) Cal State Northridge and Ohio State 2011 Details University Park , Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26 -- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park , Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018 Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion
  Fact 1.0: In 1970, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.1: In 1970, UCLA (24-1) won the final 3-0 over runner-up Long Beach State.
  Fact 1.2: In 1970, third place went to UC Santa Barbara, with a third-place score of 2-0.
  Fact 1.3: In 1970, fourth place went to Ball State.
  Fact 1.4: In 1971, UCLA (29-1) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.5: In 1971, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.6: In 1971, fourth place went to Springfield.
  Fact 1.7: In 1972, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.8: In 1972, UCLA (27-7) won the final 3-2 over runner-up San Diego State.
  Fact 1.9: In 1972, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.10: In 1972, fourth place went to UC Santa Barbara.
  Fact 1.11: In 1973, the championship was held in San Diego (host San Diego State) at Peterson Gym.
  Fact 1.12: In 1973, San Diego State (21-5) won the final 3-1 over runner-up Long Beach State.
  Fact 1.13: In 1973, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.14: In 1973, fourth place went to Army.
  Fact 1.15: In an unlabeled row after 1973, the championship was held in Santa Barbara, California (host UCSB) at Robertson Gymnasium.
  Fact 1.16: In that unlabeled row, UCLA (30-5) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.17: In that unlabeled row, third place went to Ball State, with a third-place score of 2-1.
  Fact 1.18: In that unlabeled row, fourth place went to Springfield.
  Fact 1.19: In an unlabeled row after the Santa Barbara row, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.20: In that unlabeled row, UCLA (27-8) won the final 3-1 over runner-up UC Santa Barbara.
  Fact 1.21: In that unlabeled row, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.22: In that unlabeled row, fourth place went to Yale.
  Fact 1.23: In 1976, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.24: In 1976, UCLA (15-2) won the final 3-0 over runner-up Pepperdine.
  Fact 1.25: In 1976, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.26: In 1976, fourth place went to Springfield.
  Fact 1.27: In 1977, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.28: In 1977, USC (18-1) won the final 3-1 over runner-up Ohio State.
  Fact 1.29: In 1977, third place went to Pepperdine, with a third-place score of 2-0.
  Fact 1.30: In 1977, fourth place went to Rutgers-Newark.
  Fact 1.31: In 1978, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.32: In 1978, Pepperdine (21-4) won the final 3-2 over runner-up UCLA.
  Fact 1.33: In 1978, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.34: In 1978, fourth place went to Rutgers-Newark.
  Fact 1.35: In 1979, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.36: In 1979, UCLA (30-0) won the final 3-1 over runner-up USC.
  Fact 1.37: In 1979, third place went to Rutgers-Newark, with a third-place score of 3-2.
  Fact 1.38: In 1979, fourth place went to Ball State.
  Fact 1.39: In 1980, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.40: In 1980, USC (22-6) won the final 3-1 over runner-up UCLA.
  Fact 1.41: In 1980, third place went to Ohio State, with a third-place score of 3-0.
  Fact 1.42: In 1980, fourth place went to Rutgers-Newark.
  Fact 1.43: In 1981, the championship was held in Santa Barbara, California (host UCSB) at the UCSB Events Center.
  Fact 1.44: In 1981, UCLA (32-3) won the final 3-2 over runner-up USC.
  Fact 1.45: In 1981, third place went to Penn State, with a third-place score of 3-1.
  Fact 1.46: In 1981, fourth place went to Ohio State.
  Fact 1.47: In 1982, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.48: In 1982, UCLA (29-0) won the final 3-0 over runner-up Penn State.
  Fact 1.49: In 1982, third place went to USC, with a third-place score of 2-1.
  Fact 1.50: In 1982, fourth place went to Ohio State.
  Fact 1.51: In an unlabeled row, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.52: In that unlabeled row, UCLA (27-4) won the final 3-0 over runner-up Pepperdine.
  Fact 1.53: In that unlabeled row, third place went to Ohio State, with a third-place score of 3-1.
  Fact 1.54: In that unlabeled row, fourth place went to Penn State.
  Fact 1.55: In 1984, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.56: In 1984, UCLA (38-0) won the final 3-1 over runner-up Pepperdine.
  Fact 1.57: In 1984, third place went to George Mason, with a third-place score of 3-0.
  Fact 1.58: In 1984, fourth place went to Ball State.
  Fact 1.59: In 1985, Pepperdine (25-2) won the final 3-2 over runner-up USC.
  Fact 1.60: In 1985, third place went to George Mason, with a third-place score of 3-0.
  Fact 1.61: In 1985, fourth place went to Ball State.
  Fact 1.62: In 1986, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.63: In 1986, Pepperdine (22-7) won the final 3-2 over runner-up USC.
  Fact 1.64: In 1986, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.65: In 1986, fourth place went to Ohio State.
  Fact 1.66: In an unlabeled row, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.67: In that unlabeled row, UCLA (28-3) won the final 3-0 over runner-up USC.
  Fact 1.68: In that unlabeled row, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.69: In that unlabeled row, fourth place went to Ohio State.
  Fact 1.70: In 1988, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.71: In 1988, USC (34-4) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.72: In 1988, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.73: In 1988, fourth place went to George Mason.
  Fact 1.74: In 1989, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.75: In 1989, UCLA (29-5) won the final 3-1 over runner-up Stanford.
  Fact 1.76: In 1989, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.77: In 1989, fourth place went to Ball State.
  Fact 1.78: In 1990, the championship was held in Fairfax, Virginia (host George Mason) at Patriot Center.
  Fact 1.79: In 1990, USC (26-7) won the final 3-1 over runner-up Long Beach State.
  Fact 1.80: In 1990, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.81: In 1990, fourth place went to Rutgers-Newark.
  Fact 1.82: In 1991, the championship was held in Honolulu, HI (host Hawaii) at Neal S. Blaisdell Center.
  Fact 1.83: In 1991, Long Beach State (31-4) won the final 3-1 over runner-up USC.
  Fact 1.84: In 1991, third place went to IPFW, with a third-place score of 3-1.
  Fact 1.85: In 1991, fourth place went to Penn State.
  Fact 1.86: In 1992, the championship was held in Muncie, Indiana (host Ball State) at John E. Worthen Arena.
  Fact 1.87: In 1992, Pepperdine (24-4) won the final 3-0 over runner-up Stanford.
  Fact 1.88: In 1992, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.89: In 1992, fourth place went to IPFW.
  Fact 1.90: In 1993, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.91: In 1993, UCLA (24-3) won the final 3-0 over runner-up Cal State Northridge.
  Fact 1.92: In 1993, third place went to Penn State, with a third-place score of 3-2.
  Fact 1.93: In 1993, fourth place went to Ohio State.
  Fact 1.94: In 1994, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.95: In 1994, Penn State (26-3) won the final 3-2 over runner-up UCLA.
  Fact 1.96: In 1994, third place went to Ball State, with a third-place score of 3-0.
  Fact 1.97: In 1994, fourth place went to IPFW.
  Fact 1.98: In 1995, the championship was held in Springfield, Massachusetts (host Springfield) at Springfield Civic Center.
  Fact 1.99: In 1995, UCLA (31-1) won the final 3-0 over runner-up Penn State.
  Fact 1.100: In 1995, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.101: In 1995, fourth place went to Hawaii.
  Fact 1.102: In an unlabeled row between 1995 and 1997, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.103: In that unlabeled row, UCLA (26-5) won the final 3-2 over runner-up Hawaii.
  Fact 1.104: In that unlabeled row, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.105: In 1997, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.106: In 1997, Stanford (27-4) won the final 3-2 over runner-up UCLA.
  Fact 1.107: In 1997, Ball State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.108: In 1998, the championship was held in Honolulu, HI (host Hawaii) at Stan Sheriff Center.
  Fact 1.109: In 1998, UCLA (28-4) won the final 3-0 over runner-up Pepperdine.
  Fact 1.110: In 1998, Lewis and Princeton are listed as the remaining finishers, with no score.
  Fact 1.111: In 1999, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.112: In 1999, BYU (30-1) won the final 3-0 over runner-up Long Beach State.
  Fact 1.113: In 1999, IPFW and Penn State are listed as the remaining finishers, with no score.
  Fact 1.114: In 2000, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.115: In 2000, UCLA (29-5) won the final 3-0 over runner-up Ohio State.
  Fact 1.116: In 2000, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.117: In 2001, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.118: In 2001, BYU (23-4) won the final 3-0 over runner-up UCLA.
  Fact 1.119: In 2001, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.120: In 2002, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.121: In 2002, Hawaii (24-8), marked with a dagger, won the final 3-1 over runner-up Pepperdine.
  Fact 1.122: In 2002, Ball State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.123: In 2003, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.124: In 2003, Lewis (29-6), marked with a dagger, won the final 3-2 over runner-up BYU.
  Fact 1.125: In 2003, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.126: In an unlabeled row between 2003 and 2005, the championship was held in Honolulu, HI (host Hawaii) at Stan Sheriff Center.
  Fact 1.127: In that unlabeled row, BYU (29-4) won the final 3-2 over runner-up Long Beach State.
  Fact 1.128: In that unlabeled row, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.129: In 2005, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.130: In 2005, Pepperdine (25-2) won the final 3-2 over runner-up UCLA.
  Fact 1.131: In 2005, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.132: In 2006, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.133: In 2006, UCLA (26-12) won the final 3-0 over runner-up Penn State.
  Fact 1.134: In 2006, UC Irvine and IPFW are listed as the remaining finishers, with no score.
  Fact 1.135: In 2007, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.136: In 2007, UC Irvine (29-5) won the final 3-1 over runner-up IPFW.
  Fact 1.137: In 2007, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.138: In 2008, the championship was held in Irvine, California (host UC Irvine) at Bren Events Center.
  Fact 1.139: In 2008, Penn State (30-1) won the final 3-1 over runner-up Pepperdine.
  Fact 1.140: In 2008, Long Beach State and Ohio State are listed as the remaining finishers, with no score.
  Fact 1.141: In 2009, the championship was held in Provo, Utah (host BYU) at Smith Fieldhouse.
  Fact 1.142: In 2009, UC Irvine (27-5) won the final 3-2 over runner-up USC (21-11).
  Fact 1.143: In 2009, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.144: In an unlabeled row between 2009 and 2011, the championship was held in Stanford, California (host Stanford) at Maples Pavilion.
  Fact 1.145: In that unlabeled row, Stanford (24-6) won the final 3-0 over runner-up Penn State (24-8).
  Fact 1.146: In that unlabeled row, Cal State Northridge and Ohio State are listed as the remaining finishers, with no score.
  Fact 1.147: In 2011, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.148: In 2011, Ohio State (26-6) won the final 3-2 over runner-up UC Santa Barbara (18-15).
  Fact 1.149: In 2011, Penn State and USC are listed as the remaining finishers, with no score.
  Fact 1.150: In 2012, the championship was held in Los Angeles (host USC) at Galen Center.
  Fact 1.151: In 2012, UC Irvine (26-5) won the final 3-0 over runner-up USC (24-6).
  Fact 1.152: In 2012, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.153: In 2013, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.154: In 2013, UC Irvine (25-7) won the final 3-0 over runner-up BYU (26-5).
  Fact 1.155: In 2013, Loyola Chicago and Penn State are listed as the remaining finishers, with no score.
  Fact 1.156: In 2014, the championship was held in Chicago (host Loyola Chicago) at Gentile Arena.
  Fact 1.157: In 2014, Loyola Chicago (29-1) won the final 3-1 over runner-up Stanford (24-9).
  Fact 1.158: In 2014, BYU and Penn State are listed as third place.
  Fact 1.159: In 2014, Lewis and Erskine are listed as fifth place.
  Fact 1.160: In 2015, the championship was held in Stanford, California (host Stanford) at Maples Pavilion.
  Fact 1.161: In 2015, Loyola Chicago (28-2) won the final 3-2 over runner-up Lewis (27-4).
  Fact 1.162: In 2015, UC Irvine and Penn State are listed as third place.
  Fact 1.163: In 2015, Hawaii and Pfeiffer are listed as fifth place.
  Fact 1.164: In 2016, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.165: In 2016, Ohio State (31-2) won the final 3-0 over runner-up BYU (27-4).
  Fact 1.166: In 2016, UCLA and Long Beach State are listed as third place.
  Fact 1.167: In 2016, Erskine and George Mason are listed as fifth place.
  Fact 1.168: In 2017, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.169: In 2017, Ohio State (32-2) won the final 3-0 over runner-up BYU (26-5).
  Fact 1.170: In 2017, Hawaii and Long Beach State are listed as third place.
  Fact 1.171: In 2017, Barton and Penn State are listed as fifth place.
  Fact 1.172: In 2018, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.173: In 2019, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.174: In 2020, the championship was held in Fairfax, Virginia (host George Mason) at EagleBank Arena.
  Fact 1.175: In 2021, the championship was held in Columbus, Ohio (host Ohio State) at Covelli Arena.
  Fact 1.176: In 2022, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.

Retrieved items:
<item 1>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details
Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018
Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George
Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion † Vacated due to NCAA
violations Team titles ( edit ) UCLA Pepperdine USC UC Irvine BYU Penn State Stanford Loyola Ohio State ⇙ Hawaii Lewis Long Beach State San Diego State Schools
with national championships -- 19 championships , -- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships ,
</item 1>

<item 2>
NCAA Men's National Collegiate Volleyball Championship
Indiana ( IPFW ) ACWMC Penn State ( 26 -- 3 ) 3 -- 2 UCLA Ball State 3 -- 0 IPFW 1995 Details Springfield , Massachusetts ( Springfield ) Springfield Civic
Center UCLA ( 31 -- 1 ) 3 -- 0 Penn State Ball State 3 -- 1 Hawaii Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 26 -- 5 ) 3 -- 2 Hawaii Lewis and Penn
State 1997 Details Columbus , Ohio ( Ohio State ) St. John Arena Stanford ( 27 -- 4 ) 3 -- 2 UCLA Ball State and Penn State 1998 Details Honolulu , HI ( Hawaii
) Stan Sheriff Center UCLA ( 28 -- 4 ) 3 -- 0 Pepperdine Lewis and Princeton 1999 Details Los Angeles ( UCLA ) Pauley Pavilion BYU ( 30 -- 1 ) 3 -- 0 Long Beach
State IPFW and Penn State 2000 Details Fort Wayne , Indiana ( IPFW ) ACWMC UCLA ( 29 -- 5 ) 3 -- 0 Ohio State Penn State and Pepperdine 2001 Details Long Beach
,
</item 2>

<item 3>
NCAA Men's National Collegiate Volleyball Championship
Indiana ( IPFW ) ACWMC USC ( 34 -- 4 ) 3 -- 2 UC Santa Barbara Ball State 3 -- 1 George Mason 1989 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 29 -- 5 )
3 -- 1 Stanford Penn State 3 -- 0 Ball State 1990 Details Fairfax , Virginia ( George Mason ) Patriot Center USC ( 26 -- 7 ) 3 -- 1 Long Beach State Ball State
3 -- 1 Rutgers -- Newark 1991 Details Honolulu , HI ( Hawaii ) Neal S. Blaisdell Center Long Beach State ( 31 -- 4 ) 3 -- 1 USC IPFW 3 -- 1 Penn State 1992
Details Muncie , Indiana ( Ball State ) John E. Worthen Arena Pepperdine ( 24 -- 4 ) 3 -- 0 Stanford Penn State 3 -- 0 IPFW 1993 Details Los Angeles ( UCLA )
Pauley Pavilion UCLA ( 24 -- 3 ) 3 -- 0 Cal State Northridge Penn State 3 -- 2 Ohio State 1994 Details Fort Wayne ,
</item 3>

<item 4>
NCAA Men's National Collegiate Volleyball Championship
California - based universities have dominated the men 's volleyball national championship ; Loyola Chicago , Penn State , Ohio State and BYU are the only
non-California universities to have won the National Collegiate championship ; Hawaii and Lewis also won the championship tournament , but had their victories
vacated due to NCAA rules violations . Only seven non-California universities have participated in the National Collegiate championship match ( Loyola , BYU ,
Penn State , Ohio State , IPFW , Hawaii , and Lewis ) , although other universities such as Princeton and Ball State have participated in the final four . Only
three finals have involved two non-California schools : the 2003 final , when Lewis defeated BYU but had its win vacated ; the 2015 final , in which Loyola
defeated Lewis ; and the 2016 final , when Ohio State defeated BYU . UCLA , Southern California , Penn State , Stanford , and Long Beach State are the only
schools in Division I to have won an NCAA national championship in both men and women 's volleyball .
</item 4>

<item 5>
NCAA Men's National Collegiate Volleyball Championship
-- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships , -- 1 championship School # Year won UCLA 19 1970 , 1971 , 1972 , 1974 , 1975
, 1976 , 1979 , 1981 , 1982 , 1983 , 1984 , 1987 , 1989 , 1993 , 1995 , 1996 , 1998 , 2000 , 2006 Pepperdine 5 1978 , 1985 , 1986 , 1992 , 2005 Southern
California 1977 , 1980 , 1988 , 1990 UC Irvine 2007 , 2009 , 2012 , 2013 Ohio State 2011 , 2016 , 2017 BYU 1999 , 2001 , 2004 Penn State 1994 , 2008 Stanford
1997 , 2010 Loyola Chicago 2014 , 2015 Vacated 2002 , 2003 Long Beach State 1991 San Diego State 1973 Past tournaments ( edit ) Historically ,
</item 5>

<item 6>
NCAA Men's National Collegiate Volleyball Championship
Stanford , California Championship Saturday , May 9 , 2015 Maples Pavilion , Stanford , California Lewis ( 3 ) 25 22 25 25 Hawaii ( 1 ) 22 20 25 25 5 Penn State
( 1 ) 20 25 16 20 5 Penn State ( 3 ) 25 25 17 27 Lewis ( 2 ) 25 23 15 27 21 Loyola ( Chicago ) ( 3 ) 21 25 25 25 23 UC Irvine ( 0 ) 22 19 17 Loyola ( Chicago )
( 3 ) 25 33 25 Loyola ( Chicago ) ( 3 ) 25 25 25 6 Pfeiffer ( 0 ) 20 31 15 2016 ( edit ) Main article : 2016 NCAA Men 's National Collegiate Volleyball
Tournament See also ( edit ) NCAA Division III Men 's Volleyball Championship NCAA Women 's Volleyball Championship List of NCAA men 's volleyball schools
Pre-NCAA Intercollegiate Volleyball Champions Notes ( edit ) Jump up ^ Athletic Business ,
</item 6>

<item 7>
NCAA Men's National Collegiate Volleyball Championship
( But for Hawaii 's vacated 2002 title , it would have matched this feat with 3 NCAA and 1 AIAW women 's national championships . ) In addition , Stanford (
1996 -- 97 ) and Penn State ( 2007 -- 08 ) are the only universities whose men and women 's volleyball programs won the national championship in the same
academic year . The 2016 championship tournament took place from May 3 -- 7 at Rec Hall on the campus of Penn State . 2011 ( edit ) May 5 , 2011 -- UC Santa
Barbara def . Southern California , 29 - 27 , 24 - 26 , 25 - 15 , 25 - 18 ; Ohio State def . Penn St. , 25 - 18 , 24 - 26 , 25 - 22 , 25 - 23 May 7 , 2011 --
Ohio State def .
</item 7>

<item 8>
NCAA Men's National Collegiate Volleyball Championship
California ( Long Beach State ) The Pyramid BYU ( 23 -- 4 ) 3 -- 0 UCLA Ohio State and Penn State 2002 Details University Park , Pennsylvania ( Penn State ) Rec
Hall Hawaii ( 24 -- 8 ) † 3 -- 1 Pepperdine Ball State and Penn State 2003 Details Long Beach , California ( Long Beach State ) The Pyramid Lewis ( 29 -- 6 ) †
3 -- 2 BYU Penn State and Pepperdine Details Honolulu , HI ( Hawaii ) Stan Sheriff Center BYU ( 29 -- 4 ) 3 -- 2 Long Beach State Lewis and Penn State 2005
Details Los Angeles ( UCLA ) Pauley Pavilion Pepperdine ( 25 -- 2 ) 3 -- 2 UCLA Ohio State and Penn State 2006 Details University Park , Pennsylvania ( Penn
State ) Rec Hall UCLA ( 26 -- 12 ) 3 -- 0 Penn State UC Irvine and IPFW 2007 Details Columbus , Ohio ( Ohio State ) St. John Arena UC Irvine ( 29 -- 5 ) 3 -- 1
IPFW Penn State and Pepperdine 2008 Details Irvine ,
</item 8>

<item 9>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen
Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26
-- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and
Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC
Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park ,
</item 9>

<item 10>
NCAA Men's National Collegiate Volleyball Championship
Champions ( edit ) NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner
Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2
-- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving
Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State (
21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara ,
</item 10>
=== END ITEM id=5a3d524195c25268 ===

=== ITEM id=f36d2c51727c11d6 ===
Gold sentences and facts:
Sentence 0: The ability to compare ammunition is a direct result of the invention of rifling around the turn of the 16th century . By forcing the bullet to spin as it travels down the barrel of the weapon the bullet 's accuracy greatly increases . At the same time , the rifling leaves marks on the bullet that are indicative of that particular barrel . Prior to mass production of firearms , each barrel and bullet mold was hand made by gunsmiths making them unique . The first successful documented case of forensic firearm examination occurred in 1835 when a member of the Bow Street Runners in London matched a recovered bullet from a murder victim to a specific mold in a suspect 's home confirming that he made the bullet . Further evidence that the bullet maker was the perpetrator was found in his home and he was convicted . As manufacturing and automation replaced hand tools , the ability to compare bullets became impossible due to the standardization of molds within a specific company . However , experts in the field postulated that there were microscopic differences on each barrel left during the manufacturing process . These differences were a result of wear on the machines and since each new weapon caused a tiny amount of wear , each barrel would be slightly different from every other barrel produced by that company . Also , each bullet fired from a specific barrel would be printed with the same marks , allowing investigators to identify the weapon that fired a specific bullet .
  Fact 0.0: The ability to compare ammunition is a direct result of the invention of rifling.
  Fact 0.1: Rifling was invented around the turn of the 16th century.
  Fact 0.2: Rifling forces the bullet to spin as it travels down the barrel of the weapon.
  Fact 0.3: Rifling greatly increases the bullet's accuracy.
  Fact 0.4: At the same time, rifling leaves marks on the bullet that are indicative of that particular barrel.
  Fact 0.5: Before mass production of firearms, each barrel and bullet mold was hand made by gunsmiths, making each one unique.
  Fact 0.6: The first successful documented case of forensic firearm examination occurred in 1835.
  Fact 0.7: In that 1835 case, a member of the Bow Street Runners in London matched a recovered bullet from a murder victim to a specific mold in a suspect's home.
  Fact 0.8: In that 1835 case, the match confirmed that the suspect made the bullet.
  Fact 0.9: Further evidence that the bullet maker was the perpetrator was found in his home.
  Fact 0.10: The bullet maker was convicted.
  Fact 0.11: As manufacturing and automation replaced hand tools, the ability to compare bullets became impossible.
  Fact 0.12: The ability to compare bullets became impossible due to the standardization of molds within a specific company.
  Fact 0.13: Experts in the field postulated that there were microscopic differences on each barrel left during the manufacturing process.
  Fact 0.14: These differences resulted from wear on the machines.
  Fact 0.15: Since each new weapon caused a tiny amount of wear, each barrel would be slightly different from every other barrel produced by that company.
  Fact 0.16: Each bullet fired from a specific barrel would be printed with the same marks.
  Fact 0.17: The same marks allow investigators to identify the weapon that fired a specific bullet.

Retrieved items:
<item 1>
Stone Age
Prior to the discovery of these `` Lomekwian '' tools , the oldest known stone tools had been found at several sites at Gona , Ethiopia , on the sediments of
the paleo - Awash River , which serve to date them . All the tools come from the Busidama Formation , which lies above a disconformity , or missing layer ,
which would have been from 2.9 to 2.7 mya . The oldest sites containing tools are dated to 2.6 -- 2.55 mya . One of the most striking circumstances about these
sites is that they are from the Late Pliocene , where previous to their discovery tools were thought to have evolved only in the Pleistocene . Excavators at the
locality point out that : `` ... the earliest stone tool makers were skilled flintknappers ... The possible reasons behind this seeming abrupt transition from
the absence of stone tools to the presence thereof include ... gaps in the geological record . '' The species who made the Pliocene tools remains unknown .
</item 1>

<item 2>
The earliest known stone tools come from Lomekwi 3, Kenya, dating to 3.3 million years ago; their makers are unknown. Better-known Oldowan tools, named for
Olduvai Gorge in Tanzania, were made by striking river pebbles with a hammerstone. This “core-and-flake” technique produced cores and sharp-edged flakes, though
the terms can vary; splitting a stone in half is called bipolar flaking.
</item 2>

<item 3>
Stone Age
Starting in the grasslands of the rift , Homo erectus , the predecessor of modern humans , found an ecological niche as a tool - maker and developed a
dependence on it , becoming a `` tool equipped savanna dweller . '' Stone Age in archaeology Beginning of the Stone Age Obsidian projectile point The oldest
indirect evidence found of stone tool use is fossilised animal bones with tool marks ; these are 3.4 million years old and were found in the Lower Awash Valley
in Ethiopia . Archaeological discoveries in Kenya in 2015 , identifying possibly the oldest known evidence of hominin use of tools to date , have indicated that
Kenyanthropus platyops ( a 3.2 to 3.5 - million - year - old Pliocene hominin fossil discovered in Lake Turkana , Kenya in 1999 ) may have been the earliest
tool - users known . The oldest stone tools were excavated from the site of Lomekwi 3 in West Turkana , northwestern Kenya , and date to 3.3 million years old .
</item 3>

<item 4>
Acheulean Mode 2 tools first appear outside Africa at ‘Ubeidiya, Israel, where Homo occupied the shore of a now-vanished, fluctuating palaeolake for hundreds of
thousands of years. Tools occur in the lake’s first two geological cycles, mainly the fluviatile Fi cycle. Faunal evidence links these layers to Italy’s Ferenta
Faunal Unit, dated 1.6–1.2 million years ago. Bone marks suggest toolmakers scaveng
</item 4>

<item 5>
Stone Age
This is a Mode 1 , or Oldowan , stone tool from the western Sahara . Oldowan in Africa Main article : Oldowan The earliest documented stone tools have been
found in eastern Africa , manufacturers unknown , at the 3.3 million year old site of Lomekwi 3 in Kenya . Better known are the later tools belonging to an
industry known as Oldowan , after the type site of Olduvai Gorge in Tanzania . The tools were formed by knocking pieces off a river pebble , or stones like it ,
with a hammerstone to obtain large and small pieces with one or more sharp edges . The original stone is called a core ; the resultant pieces , flakes .
Typically , but not necessarily , small pieces are detached from a larger piece , in which case the larger piece may be called the core and the smaller pieces
the flakes . The prevalent usage , however , is to call all the results flakes , which can be confusing . A split in half is called bipolar flaking .
Consequently , the method is often called `` core - and - flake '' .
</item 5>

<item 6>
The makers of the Pliocene Gona tools are unknown; nearby fossils include several Australopithecus species and possibly Homo habilis. Stone Age societies
transitioned to metalworking at different times, generally between 6000 and 2500 BCE. Copper smelting preceded bronze production, forming the Chalcolithic or
Copper Age; bronze later gave way to iron. Early copper metallurgy is evidenced at Serbian sites around the fifth–sixth millennium BCE. Ötzi carried
</item 6>

<item 7>
Mode 2 tools are strongly associated with Eurasian H. erectus; Mode 1 links are rarer. Yiron, Israel, has Mode 1 tools dated to 2.4 million years ago—earlier
than known Eurasian erectus—raising the possibility of an earlier migrant or undiscovered early erectus. Pebble tools appeared at Gona, Ethiopia (2.7 mya), and
later across Africa and Asia, including Pakistan and China. A 1.8-m
</item 7>

<item 8>
Stone Age archaeology studies how tools were made through lithic analysis and experimental replicas, alongside evidence from geology, fossils, pollen, and
scientific dating. Its focus is ultimately the people and societies, not tools alone. Stone tools included chipped flint and chert and ground basalt and
sandstone; people also used wood, bone, shell, and antler. Later, pottery, agriculture, and animal domestication emerged. The period spans cultures associated
with the Upper Paleolithic, Mesolithic
</item 8>

<item 9>
Stone Age
There is additional evidence that Oldowan , or Mode 1 , tools were utilized in `` percussion technology '' ; that is , they were designed to be gripped at the
blunt end and strike something with the edge , from which use they were given the name of choppers . Modern science has been able to detect mammalian blood
cells on Mode 1 tools at Sterkfontein , Member 5 East , in South Africa . As the blood must have come from a fresh kill , the tool users are likely to have done
the killing and used the tools for butchering . Plant residues bonded to the silicon of some tools confirm the use to chop plants . Although the exact species
authoring the tools remains unknown , Mode 1 tools in Africa were manufactured and used predominantly by Homo habilis . They can not be said to have developed
these tools or to have contributed the tradition to technology . They continued a tradition of yet unknown origin . As chimpanzees sometimes naturally use
percussion to extract or prepare food in the wild , and may use either unmodified stones or stones that they have split , creating an Oldowan tool , the
tradition may well be far older than its current record .
</item 9>

<item 10>
Stone Age
Some Oldowan tools are more carefully prepared to form a more regular edge . One distinguishing criterion is the size of the flakes . In contrast to the Oldowan
`` small flake '' tradition , Acheulean is `` large flake : '' `` The primary technological distinction remaining between Oldowan and the Acheulean is the
preference for large flakes ( > 10 cm ) as blanks for making large cutting tools ( handaxes and cleavers ) in the Acheulean . '' `` Large Cutting Tool ( LCT )
'' has become part of the standard terminology as well . In North Africa , the presence of Mode 2 remains a mystery , as the oldest finds are from Thomas Quarry
in Morocco at 0.9 mya . Archaeological attention , however , shifts to the Jordan Rift Valley , an extension of the East African Rift Valley ( the east bank of
the Jordan is slowly sliding northward as East Africa is thrust away from Africa ) . Evidence of use of the Nile Valley is in deficit , but Hominans could
easily have reached the palaeo - Jordan river from Ethiopia along the shores of the Red Sea , one side or the other .
</item 10>
=== END ITEM id=f36d2c51727c11d6 ===

=== ITEM id=082a7e9ac9ad21ab ===
Gold sentences and facts:
Sentence 0: The phrase `` objects in ( the ) mirror are closer than they appear '' is a safety warning that is required to be engraved on passenger side mirrors of motor vehicles in the USA , Canada , Nepal , India and Saudi Arabia . It is present because while these mirrors ' convexity gives them a useful field of view , it also makes objects appear smaller . Since smaller - appearing objects seem farther away than they actually are , a driver might make a maneuver such as a lane change assuming an adjacent vehicle is a safe distance behind , when in fact it is quite a bit closer . The warning serves as a reminder to the driver of this potential problem .
  Fact 0.0: The phrase 'objects in (the) mirror are closer than they appear' is a safety warning.
  Fact 0.1: The safety warning is required to be engraved on passenger side mirrors of motor vehicles.
  Fact 0.2: The engraving requirement applies in the USA, Canada, Nepal, India and Saudi Arabia.
  Fact 0.3: The warning is present on passenger side mirrors because their convexity gives them a useful field of view.
  Fact 0.4: The convexity of passenger side mirrors makes objects appear smaller.
  Fact 0.5: Smaller-appearing objects seem farther away than they actually are.
  Fact 0.6: A driver might make a maneuver such as a lane change assuming an adjacent vehicle is a safe distance behind.
  Fact 0.7: In fact, the adjacent vehicle is quite a bit closer than the driver assumes.
  Fact 0.8: The warning serves as a reminder to the driver of this potential problem.

Retrieved items:
<item 1>
If a tree falls in a forest
If there be no ears to hear , there will be no sound . '' Albert Einstein is reported to have asked his fellow physicist and friend Niels Bohr , one of the
founding fathers of quantum mechanics , whether he realistically believed that ' the moon does not exist if nobody is looking at it . ' To this Bohr replied
that however hard he ( Einstein ) may try , he would not be able to prove that it does , thus giving the entire riddle the status of a kind of an infallible
conjecture -- one that can not be either proved or disproved . The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann
and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed
along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view . Metaphysics (
edit ) The possibility of unperceived existence ( edit ) Main article : George Berkeley Can something exist without being perceived ? -- e.g.
</item 1>

<item 2>
Objects in mirror are closer than they appear
A reference in the 2011 film Transformers : Dark of the Moon film occurs when Dino , Bumblebee and Sideswipe are protecting Sentinel from the dreads , this
appears in a brief shot in Dino 's mirror . `` Objects In The Mirror , '' a song performed by American rapper Mac Miller on his second studio album Watching
Movies with the Sound Off `` Subjects in the mirror are more far than they appear '' the title of experimental visual project by Hungarian director / production
designer Pater Sparrow `` Objects in the Mirror , '' a 2015 song by a-ha . Objects in the mirror , a song performed by Chris Miles . ( 2016 ) Objects in mirror
are closer than they appear was a level in the videogame `` Bugs Bunny : Lost in time '' ( PlayStation 1 ) . In My Little Pony : The Movie , the sentence ``
Objects in windows may be less mountainous than they appear '' is uttered .
</item 2>

<item 3>
Objects in mirror are closer than they appear
Explanation from PhysLink.com . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Objects_in_mirror_are_closer_than_they_appear&oldid=838121569 ''
Categories : English phrases Mirrors Car safety Hidden categories : Wikipedia articles needing clarification from March 2016 Interlanguage link template link
number Talk About Wikipedia 한국어 Edit links This page was last edited on 25 April 2018 , at 01 : 58 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Objects in mirror are closer than they appear
In popular culture ( edit ) Despite its origin as a utilitarian safety warning , the phrase has become a well known catch phrase that has been used for many
other purposes . Some of them are : Objects in the Mirror Are Closer Than They Appear , the 2000 debut album of rap rock group Confrontation Camp . Objects in
the Mirror Are Closer Than They Appear , a 1993 album by Nancy Moran Objects in Mirror Are Closer Than They Appear , a 2005 album by Jozef van Wissem . Objects
in Mirror Are Closer Than They Appear , a 1995 novel by Katharine Weber . Showing an approach of a Tyrannosaurus rex in the film Jurassic Park , which was
parodied in Toy Story 2 . A Far Side cartoon by Gary Larson in which the side mirror showed a very large eyeball . A reference in Ace Ventura : Pet Detective ,
in which a pejorative term replaces `` objects . '' `` Objects in the Rear View Mirror May Appear Closer than They Are , '' a 1994 song performed by Meat Loaf
and written by Jim Steinman - the opposite of the safety warning .
</item 4>

<item 5>
If a tree falls in a forest
Washington - state - based wilderness conservatory Northwest Trek used a shortened form of the quote in its mid-1970s television advertisement , as such : ``
There is no sound unless someone is there to see it or hear it . Experience it at Northwest Trek . '' See also ( edit ) Counterfactual definiteness Descartes '
dream argument Epistemology Object ( philosophy ) Observer effect ( physics ) Kōan Ontology Schrödinger 's cat Principle of locality References ( edit ) Jump up
^ A Treatise Concerning the Principles of Human Knowledge , 1734 . section 23 . Jump up ^ A Treatise Concerning the Principles of Human Knowledge , 1734 .
section 45 . Jump up ^ Whether Berkeley thought objects continued to exist when unperceived by finite minds , and if so , in what manner , is the subject of
serious debate among Berkeley scholars . Jump up ^ The Chautauquan , June 1883 , Volume 3 , Issue 9 , p. 543 Jump up ^ Scientific American , April 5 , 1884 , pg
218 .
</item 5>

<item 6>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 6>

<item 7>
If a tree falls in a forest
It may then be recognized that the unobserved event was absolutely identical to an event which did not occur at all . '' ( this apparent quote has no
attribution or reference and none can be found online with reasonable effort ) . Of course , the fact that the tree is known to have changed state from '
upright ' to ' fallen ' implies that the event must be observed to ask the question at all -- even if only by the supposed deaf onlooker . The British
philosopher of science Roy Bhaskar , credited with developing critical realism has argued , in apparent reference to this riddle , that : If men ceased to exist
sound would continue to travel and heavy bodies to fall to the earth in exactly the same way , though ex hypothesi there would be no - one to know it This
existence of an unobserved real is integral to Bhaskar 's ontology , which contends ( in opposition to the various strains of positivism which have dominated
both natural and social science in the twentieth century ) that ' real structures exist independently of and are often out of phase with the actual patterns of
events ' .
</item 7>

<item 8>
Graph query row:
entity: Objects in mirror are closer than they appear
entity_types: ['__KGBuilder__', 'Concept', '__Entity__']
relationship: RELATED_TO
related: Federal Motor Vehicle Safety Standards
related_types: ['__KGBuilder__', 'Concept', '__Entity__']
</item 8>

<item 9>
Objects in mirror are closer than they appear
Iranian film Objects in Mirror ( fa ) See also ( edit ) Blind spot ( automobile ) Blind spot monitor Driver visibility Parking sensors Rear - view mirror Mind
the gap , another safety warning used at various London Underground stations that has also become a well - known catchphrase . References ( edit ) Jump up ^ For
example , in the U.S , PART 571 Federal Motor Vehicle Safety Standards , Section 571.111 S5. 4.2 `` Each convex mirror shall have permanently and indelibly
marked at the lower edge of the mirror 's reflective surface , in letters not less than 4.8 mm nor more than 6.4 mm high the words `` Objects in Mirror Are
Closer Than They Appear . '' Jump up ^ Why does the passenger side window on my car state ' objects in mirror are closer than they appear ? ' Explanation from
PhysLink.com .
</item 9>

<item 10>
Objects in mirror are closer than they appear
Objects in mirror are Closer than they appear - wikipedia Objects in mirror are Closer than they appear Jump to : navigation , search Side - view mirror with
legend Wing mirror on Korean - specification vehicle . Legend in Korean reads `` Objects in mirror are closer than they appear '' . Main article : Wing mirror
The phrase `` objects in ( the ) mirror are closer than they appear '' is a safety warning that is required to be engraved on passenger side mirrors of motor
vehicles in the USA , Canada , Nepal , India and Saudi Arabia . It is present because while these mirrors ' convexity gives them a useful field of view , it
also makes objects appear smaller . Since smaller - appearing objects seem farther away than they actually are , a driver might make a maneuver such as a lane
change assuming an adjacent vehicle is a safe distance behind , when in fact it is quite a bit closer . The warning serves as a reminder to the driver of this
potential problem . Comparison of the real over-the - shoulder view ( green arrow ) and apparent through the passenger side mirror ( sum of red arrows )
evaluation of distance between the driver 's head and the object behind .
</item 10>
=== END ITEM id=082a7e9ac9ad21ab ===

=== ITEM id=dd760435c4301a0c ===
Gold sentences and facts:
Sentence 0: Name Type Notes Anthropomorphism Form of personification that applies human - like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw . Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors Lear 's mental deterioration . Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . ''
  Fact 0.0: Anthropomorphism is a form of personification that applies human-like characteristics to animals and objects.
  Fact 0.1: The Adventures of Pinocchio by Carlo Collodi is an example of anthropomorphism.
  Fact 0.2: The Cheshire Cat of Alice's Adventures in Wonderland by Lewis Carroll is an example of anthropomorphism.
  Fact 0.3: Hamartia is the character flaw of an initially rich and powerful hero that leads to his tragic downfall.
  Fact 0.4: Hamartia is also referred to as the tragic flaw.
  Fact 0.5: Oedipus kills his own father because he does not understand his true parentage.
  Fact 0.6: Pathetic fallacy reflects a character's, usually the protagonist's, mood in the atmosphere or inanimate objects.
  Fact 0.7: Pathetic fallacy is related to anthropomorphism and projection.
  Fact 0.8: The storm in William Shakespeare's King Lear is an example of pathetic fallacy.
  Fact 0.9: The storm in King Lear mirrors Lear's mental deterioration.
  Fact 0.10: Personification uses comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities.
  Fact 0.11: Taken from Act I, Scene II of Romeo and Juliet: "When well-appareled April on the heel / Of limping winter treads."

Retrieved items:
<item 1>
List of narrative techniques
Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II
of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . '' References ( edit ) Jump up ^ Orehovec , Barbara ( 2003 ) .
Revisiting the Reading Workshop : A Complete Guide to Organizing and Managing an Effective Reading Workshop That Builds Independent , Strategic Readers (
illustrated ed . ) . Scholastic Inc. p. 89 . ISBN 0439444047 . Jump up ^ Demchick , Harrison ( 2013 - 09 - 26 ) . `` Techniques and Tension in Breaking Bad '' .
Huffington Post . Retrieved 2017 - 11 - 14 . Jump up ^ `` Literature Glossary - Defamiliarization '' . www.shmoop.com . Retrieved 2017 - 11 - 14 . Jump up ^ ``
Allegory Examples '' . YourDictionary . Retrieved 2017 - 11 - 14 .
</item 1>

<item 2>
List of narrative techniques
If it 's not going to be fired , it should n't be hanging there . '' -- Anton Chekhov Unreliable narrator The narrator of the story is not sincere , or
introduces a bias in their narration and possibly misleads the reader , hiding or minimizing events , characters , or motivations . An example is The Murder of
Roger Ackroyd . The novel includes an unexpected plot twist at the end of the novel . In the last chapter , Sheppard describes how he was an unreliable narrator
. Perspective ( edit ) Name Definition Example Audience surrogate A character who expresses the questions and confusion of the audience , with whom the audience
can identify . Frequently used in detective fiction and science fiction , where the character asks a central character how he or she accomplished certain deeds
, for the purpose of inciting that character to explain ( for the curious audience ) his or her methods , or a character asking a relatively educated person to
explain what amounts to the backstory . Dr Watson in the Sherlock Holmes stories . Scott Evil , played by Seth Green , son of Dr. Evil on the Austin Powers
movies . Author surrogate Characters which are based on authors , usually to support their personal views .
</item 2>

<item 3>
Stranger Things
As casting was started immediately after Netflix greenlit the show , and prior to the scripts being fully completed , this allowed some of the actor 's takes on
the roles to reflect into the script . The casting of the young actors for Will and his friends had been done just after the first script was completed , and
subsequent scripts incorporated aspects from these actors . The brothers said Modine provided significant input on the character of Dr. Brenner , whom they had
not really fleshed out before as they considered him the hardest character to write for given his limited appearances within the narrative . In October 2016 ,
it was announced that Schnapp and Keery had been promoted to the main cast for the second season , after each recurring in the first season , and that Sadie
Sink and Dacre Montgomery would join the main cast as Max and Billy , respectively . Ryder , Harbour , Wolfhard , Brown , Matarazzo , McLaughlin , Dyer and
Heaton also return for the season . Sean Astin as Bob Newby and Paul Reiser as Sam Owens are also part of the main cast in the season .
</item 3>

<item 4>
Atticus Finch
This kind of character development , where motivations and ideals between characters , for reasons of plot , are changed is not unusual in the process of
creative writing . Apart from the more progressive depiction of Atticus , the depiction of the town itself , especially the African - American characters , is
also dramatically altered between the two drafts. Real - life comparisons with Lee 's father , Amasa Coleman Lee , have also been made in the two differing
versions of Atticus in that originally Amasa Lee was in favor of segregation but became more liberal later in life , later changing his views to those of
Integration . Tay Hohoff , Lee 's editor , has also been argued to have played a major part in the character development of the novel and particularly Atticus '
liberal transformation . Jonathan Mahler of The New York Times notes in his article The Invisible Hand Behind Harper Lee 's ' To Kill a Mockingbird ' that Ms.
Hohoff , at the same time as she was guiding Ms. Lee through the Mockingbird re-write , was working on her own biography of the early - 20th - century New York
activist and humanist John Lovejoy Elliot .
</item 4>

<item 5>
List of narrative techniques
Author surrogate Characters which are based on authors , usually to support their personal views . Sometimes an intentionally or unintentionally idealized
version of them . A variation is the Mary Sue or Gary Stu , which primarily serves as an idealized self - insertion . Socrates in the writings of Plato . Plato
never speaks in his own voice in his dialogues . In the Second Letter , it says , `` no writing of Plato exists or ever will exist , but those now said to be
his are those of a Socrates become beautiful and new '' . Breaking the fourth wall An author or character addresses the audience directly ( also known as direct
address ) . This may acknowledge to the reader or audience that what is being presented is fiction , or may seek to extend the world of the story to provide the
illusion that they are included in it . The characters in Sesame Street often break the fourth wall when they address their viewers as part of the ongoing
storyline , which is possible because of the high level of suspension of belief afforded by its audience -- children . The American political drama show House
of Cards also uses this technique frequently to let the viewers know what the main character Frank Underwood is thinking and planning .
</item 5>

<item 6>
List of narrative techniques
Her tears were a river flowing down her cheeks . Thematic patterning Distributing recurrent thematic concepts and moralistic motifs among various incidents and
frames of a story . In a skillfully crafted tale , thematic patterning may emphasize the unifying argument or salient idea disparate events and disparate frames
have in common . Each of the chapters of Ulysses by James Joyce . Character ( edit ) Name Type Notes Anthropomorphism Form of personification that applies human
- like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis
Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw .
Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in
the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors
Lear 's mental deterioration .
</item 6>

<item 7>
List of narrative techniques
res Pace Plot device Plot twist Poetic justice Reveal Self - fulfilling prophecy Subplot Trope Kishōtenketsu Setting Backstory Utopia Dystopia Alternate history
Fictional location city country universe Theme Leitmotif Metaphor Moral Motif Irony Style Allegory Bathos Diction Figure of speech Imagery Narrative techniques
Narration Stylistic device Suspension of disbelief Symbolism Tone Mode Mood Structure Linear narrative Nonlinear narrative films television series Types of
fiction with multiple endings Form Comics Epic Fable Fabliau Fairy tale Folktale Flash fiction Legend Novella Novel Parable Play Poem Screenplay Short story
Genre Action fiction Adventure Comic Crime Docufiction Epistolary Erotic Fiction Fantasy Gothic Historical Horror Magic realism Mystery Nautical Paranoid
Philosophical Picaresque Political Psychological Romance Saga Satire Science Speculative Superhero Thriller Urban Western List of writing genres Narration First
- person Multiple narrators Stream of consciousness Stream of unconsciousness Unreliable Tense Past Present Future Related Audience Author Creative nonfiction
Fiction writing Literary theory Literary science Narratology Monomyth Rhetoric Screenwriting Storytelling Tellability Literature portal Literary composition
General topics Literature Fiction writing Writer Characterization Exposition Description Writer 's block Techniques / devices
</item 7>

<item 8>
Drawing
The exterior is termed the negative space , and can be as important in the representation as the figure . Objects placed in the background of the figure should
appear properly placed wherever they can be viewed . A study is a draft drawing that is made in preparation for a planned final image . Studies can be used to
determine the appearances of specific parts of the completed image , or for experimenting with the best approach for accomplishing the end goal . However a well
- crafted study can be a piece of art in its own right , and many hours of careful work can go into completing a study . Process ( edit ) A person drawing the
Barberini Faun in Munich Individuals display differences in their ability to produce visually accurate drawings . A visually accurate drawing is described as
being `` recognized as a particular object at a particular time and in a particular space , rendered with little addition of visual detail that can not be seen
in the object represented or with little deletion of visual detail '' . Investigative studies have aimed to explain the reasons why some individuals draw better
than others .
</item 8>

<item 9>
List of narrative techniques
Defamiliarization Taking an everyday object and presenting it in a way that is weirdly unfamiliar so that we see the object in a new way . Coined by the early
20th - century Russian literary critic Viktor Shklovsky in `` Art as Technique . '' In , Swift 's Gulliver 's Travels when Gulliver visits the land of the
giants and sees a giant woman 's skin he sees it is anything but smooth and beautiful when viewed up close . First - person narration A text presented from the
point of view of a character , especially the protagonist , as if the character is telling the story themselves . ( Breaking the fourth wall is an option , but
not a necessity , of this format . ) Mark Twain 's Adventures of Huckleberry Finn uses the title character as the narrator , while Sherlock Holmes is primarily
told from Watson 's perspective . Magical realism Describing events in a real - world setting but with magical trappings , often incorporating local customs and
invented beliefs . Different from urban fantasy in that the magic itself is not the focus of the story . Particularly popular with Latin American authors like
Gabriel García Márquez and Jorge Luis Borges .
</item 9>

<item 10>
List of narrative techniques
This technique frequently occurs within a poetic line grammatically connected to the end of the previous line by enjambment . E.g. , in `` Know then thyself . ‖
Presume not God to scan . '' Distancing effect Deliberately preventing the audience from identifying with characters in order to let them be coolly scrutinized
. Popularized by 20th century playwright Bertolt Brecht . Dramatic visualization Representing an object or character with abundant descriptive detail , or
mimetically rendering gestures and dialogue to make a scene more visual or imaginatively present to an audience . This technique appears at least as far back as
the Arabian Nights . Euphuism An artificial , highly elaborate way of writing or speaking . Named from Euphues ( 1579 ) the prose romance by John Lyly . `` Is
it not far better to abhor sins by the remembrance of others ' faults , than by repentance of thine own follies ? '' ( Euphues , 1 , lecture by the wise
Neapolitan ) Hyperbole Exaggeration used to evoke strong feelings or create an impression which is not meant to be taken literally .
</item 10>
=== END ITEM id=dd760435c4301a0c ===

=== ITEM id=eeaa468454d42865 ===
Gold sentences and facts:
Sentence 0: Important production centres today are the Hallertau in Germany ( more hop - growing area than any other country as of 2006 ) , the Yakima ( Washington ) and Willamette ( Oregon ) valleys , and western Canyon County , Idaho ( including the communities of Parma , Wilder , Greenleaf , and Notus ) . The principal production centres in the UK are in Kent ( which produces Kent Goldings hops ) , Herefordshire , and Worcestershire . Essentially all of the harvested hops are used in beer making .
  Fact 0.0: The Hallertau in Germany has more hop-growing area than any other country as of 2006.
  Fact 0.1: The Hallertau in Germany is an important hop production centre today.
  Fact 0.2: The Yakima valley in Washington is an important hop production centre today.
  Fact 0.3: The Willamette valley in Oregon is an important hop production centre today.
  Fact 0.4: Western Canyon County, Idaho is an important hop production centre today.
  Fact 0.5: Western Canyon County, Idaho includes the communities of Parma, Wilder, Greenleaf, and Notus.
  Fact 0.6: The principal hop production centres in the UK are in Kent.
  Fact 0.7: Kent produces Kent Goldings hops.
  Fact 0.8: The principal hop production centres in the UK are in Herefordshire.
  Fact 0.9: The principal hop production centres in the UK are in Worcestershire.
  Fact 0.10: Essentially all of the harvested hops are used in beer making.
Sentence 1: Hops production is concentrated in moist temperate climates , with much of the world 's production occurring near the 48th parallel north . Hop plants prefer the same soils as potatoes and the leading potato - growing states in the United States are also major hops - producing areas ; however , not all potato - growing areas can produce good hops naturally : soils in the Maritime Provinces of Canada , for example , lack the boron that hops prefer . Historically , hops were not grown in Ireland , but were imported from England . In 1752 more than 500 tons of English hops were imported through Dublin alone .
  Fact 1.0: Hops production is concentrated in moist temperate climates.
  Fact 1.1: Much of the world's hops production occurs near the 48th parallel north.
  Fact 1.2: Hop plants prefer the same soils as potatoes.
  Fact 1.3: The leading potato-growing states in the United States are also major hops-producing areas.
  Fact 1.4: Not all potato-growing areas can produce good hops naturally.
  Fact 1.5: Soils in the Maritime Provinces of Canada lack the boron that hops prefer.
  Fact 1.6: Historically, hops were not grown in Ireland.
  Fact 1.7: Historically, hops were imported into Ireland from England.
  Fact 1.8: In 1752 more than 500 tons of English hops were imported through Dublin alone.

Retrieved items:
<item 1>
Hops are the female flower cones of the climbing perennial Humulus lupulus, used mainly to flavor and stabilize beer. Their acids add bitterness and inhibit
unwanted microbes; oils provide aroma. First cultivated in Germany by 736, hops gradually displaced herbal gruit. Commercial plants grow on trellises and are
harvested and dried; production centers include Germany, the U.S., China and the UK. Brewers use bittering, aroma or dual-purpose varieties, adding them at
different brewing stages. Noble European hops
</item 1>

<item 2>
Hops
private and public breeding programs are using their stock material . Particular hop varieties are associated with beer regions and styles , for example pale
lagers are usually brewed with European ( often German , Polish or Czech ) noble hop varieties such as Saaz , Hallertau and Strissel Spalt . British ales use
hop varieties such as Fuggles , Goldings and W.G.V. North American beers often use Cascade hops , Columbus hops , Centennial hops , Willamette , Amarillo hops
and about forty more varieties as the U.S.A. have lately been the more significant breeders of new hop varieties , including dwarf hop varieties . Hops from New
Zealand , such as Pacific Gem , Motueka and Nelson Sauvin , are used in a `` Pacific Pale Ale '' style of beer with increasing production in 2014 . Noble hops (
edit ) Mature hops growing in a hop yard in Germany The term `` noble hops '' traditionally refers to varieties of hops which are low in bitterness and high in
aroma . They are the European cultivars or races Hallertau , Tettnanger , Spalt , and Saaz .
</item 2>

<item 3>
Hops
Hops - wikipedia Hops Jump to : navigation , search For other uses , see hop ( disambiguation ) . Hop flower in a hop yard in the Hallertau , Germany
Cross-section drawing of a hop Hops are the flowers ( also called seed cones or strobiles ) of the hop plant Humulus lupulus . They are used primarily as a
flavouring and stability agent in beer , to which they impart bitter , zesty , or citric flavours ; though they are also used for various purposes in other
beverages and herbal medicine . The hop plant is a vigorous , climbing , herbaceous perennial , usually trained to grow up strings in a field called a hopfield
, hop garden ( nomenclature in the South of England ) , or hop yard ( in the West Country and U.S. ) when grown commercially . Many different varieties of hops
are grown by farmers around the world , with different types used for particular styles of beer . The first documented use of hops in beer is from the 9th
century , though Hildegard of Bingen , 300 years later , is often cited as the earliest documented source .
</item 3>

<item 4>
Hops
In 1752 more than 500 tons of English hops were imported through Dublin alone . Important production centres today are the Hallertau in Germany ( more hop -
growing area than any other country as of 2006 ) , the Yakima ( Washington ) and Willamette ( Oregon ) valleys , and western Canyon County , Idaho ( including
the communities of Parma , Wilder , Greenleaf , and Notus ) . The principal production centres in the UK are in Kent ( which produces Kent Goldings hops ) ,
Herefordshire , and Worcestershire . Essentially all of the harvested hops are used in beer making .
</item 4>

<item 5>
Hops
Hops were then grown as far north as Aberdeen , near breweries for infrastructure convenience . According to Thomas Tusser 's 1557 Five Hundred Points of Good
Husbandry : `` The hop for his profit I thus do exalt , It strengtheneth drink and it flavoureth malt ; And being well - brewed long kept it will last , And
drawing abide , if ye draw not too fast . '' In England there were many complaints over the quality of imported hops , the sacks of which were often
contaminated by stalks , sand or straw to increase their weight . As a result , in 1603 , King James I approved an Act of Parliament banning the practice by
which `` the Subjects of this Realm have been of late years abused &c. to the Value of £ 20,000 yearly , besides the Danger of their Healths '' . Hop
cultivation was begun in the present - day United States in 1629 by English and Dutch farmers . Before prohibition , cultivation was mainly centred around New
York , California , Oregon , and Washington state .
</item 5>

<item 6>
Hops
461 . Jump up ^ Charles W. Bamforth ( 1998 ) . Beer : tap into the art and science of brewing . Plenum Press . p. 245 . ISBN 0 - 306 - 45797 - 0 . Jump up ^
Knight , Paul D. `` HOPS IN BEER '' . USA Hops . Hop Growers of America . Retrieved 11 June 2015 . Jump up ^ Annual Report of the Commissioner of Patents , Part
2 ; Part 4 . US Patent office . 1858 . p. 291 . ^ Jump up to : Peicanada Jump up ^ `` The London magazine , 1752 '' , page 332 Jump up ^ Summary of Reports :
Nürnberg , Germany , 14 November 2006 , International Hop Growers ' Convention : Economic Committee Jump up ^ `` NCGR - Corvallis Humulus Genetic Resources '' .
Ars-grin.gov . Retrieved 2012 - 05 - 20 . Jump up ^ Norman Moss , A Fancy to Worcesters Archived 29 September 2007 at the Wayback Machine.
</item 6>

<item 7>
Hops
Problems with powdery mildew and downy mildew devastated New York 's production by the 1920s , and California only produces hops on a small scale . Hop bars
were used before modern machinery was invented to make the holes for the hop poles . World production ( edit ) Hops production is concentrated in moist
temperate climates , with much of the world 's production occurring near the 48th parallel north . Hop plants prefer the same soils as potatoes and the leading
potato - growing states in the United States are also major hops - producing areas ; however , not all potato - growing areas can produce good hops naturally :
soils in the Maritime Provinces of Canada , for example , lack the boron that hops prefer . Historically , hops were not grown in Ireland , but were imported
from England . In 1752 more than 500 tons of English hops were imported through Dublin alone .
</item 7>

<item 8>
Hops
Because pollinated seeds are undesirable for brewing beer , only female plants are grown in hop fields , thus preventing pollination . Female plants are
propagated vegetatively , and male plants are culled if plants are grown from seeds . Hop plants are planted in rows about 2 to 2.5 metres ( 7 to 8 ft ) apart .
Each spring , the roots send forth new bines that are started up strings from the ground to an overhead trellis . The cones grow high on the bine , and in the
past , these cones were picked by hand . Harvesting of hops became much more efficient with the invention of the mechanical hops separator , patented by Emil
Clemens Horst in 1909 . Harvest comes near the end of summer when the bines are pulled down and the flowers are taken to a hop house or oast house for drying .
Hop houses are two - story buildings , of which the upper story has a slatted floor covered with burlap . Here the flowers are poured out and raked even . A
heating unit on the lower floor is used to dry the hops .
</item 8>

<item 9>
Hops
Essentially all of the harvested hops are used in beer making . Early season hop growth in a hop yard in the Yakima River Valley of Washington with Mount Adams
in the distance Hop producing country 2012 hop output in tonnes ( t ) Germany 34,434 United States 27,782 China 11,000 Czech Republic 4,338 Poland 2,206 North
Korea 2,000 Albania 1,650 United Kingdom 1,650 Slovenia 1,338 Cultivation and harvest ( edit ) A superstructure of overhead wires supports strings that in turn
support bines Although hops are grown in most of the continental United States and Canada , cultivation of hops for commercial production requires a particular
environment . As hops are a climbing plant , they are trained to grow up trellises made from strings or wires that support the plants and allow them
significantly greater growth with the same sunlight profile . In this way , energy that would have been required to build structural cells is also freed for
crop growth . The hop plant 's reproduction method is that male and female flowers develop on separate plants , although occasionally a fertile individual will
develop which contains both male and female flowers .
</item 9>

<item 10>
Hops
A heating unit on the lower floor is used to dry the hops . When dry , the hops are moved to a press , a sturdy box with a plunger . Two long pieces of burlap
are laid into the hop press at right angles , the hops are poured in and compressed into bales . Hop cones contain different oils , such as lupulin , a
yellowish , waxy substance , an oleoresin , that imparts flavour and aroma to beer . Lupulin contains lupulone and humulone , which possess antibiotic
properties , suppressing bacterial growth favoring brewer 's yeast to grow . After lupulin has been extracted in the brewing process the papery cones are
discarded . Migrant labor and social impact ( edit ) Hops harvest in the Kingdom of Bohemia ( 1898 ) Hops harvest in Skåne , Sweden in 1937 The need for massed
labor at harvest time meant hop - growing had a big social impact . Around the world , the labor - intensive harvesting work involved large numbers of migrant
workers who would travel for the annual hop harvest .
</item 10>
=== END ITEM id=eeaa468454d42865 ===

=== ITEM id=09b407f2eefdd9b6 ===
Gold sentences and facts:
Sentence 0: On the November 19 , 1999 , episode of Millionaire , Carpenter proceeded to advance to the million - dollar question without using any lifelines . He then used his Phone - A-Friend to call his father not for help , but rather to tell him he was going to win the game . Carpenter answered the question correctly and became the show 's first millionaire . His win gave him national recognition and led to multiple talk show appearances , as well as reappearances on Millionaire itself .
  Fact 0.0: On the November 19, 1999 episode of Millionaire, Carpenter advanced to the million-dollar question.
  Fact 0.1: Carpenter advanced to the million-dollar question without using any lifelines.
  Fact 0.2: Carpenter used Phone-A-Friend to call his father.
  Fact 0.3: Carpenter did not call his father for help with the question.
  Fact 0.4: Carpenter called his father to tell him he was going to win the game.
  Fact 0.5: Carpenter answered the million-dollar question correctly.
  Fact 0.6: Carpenter became the show's first millionaire.
  Fact 0.7: Carpenter's win gave him national recognition.
  Fact 0.8: Carpenter's win led to multiple talk show appearances.
  Fact 0.9: Carpenter's win led to reappearances on Millionaire itself.

Retrieved items:
<item 1>
John Carpenter (game show contestant)
Retrieved September 6 , 2009 . ^ Jump up to : `` Who Wants to Be a Millionaire '' . Gameshow Hall of Fame . GSN . January 21 , 2007 . ^ Jump up to : Vigoda ,
Arlene ( November 22 , 1999 ) . `` Million - dollar winner untaxed by celebrity '' . USA Today . Gannett Company . p. 1D . Jump up ^ `` Taxman scoops a million
'' . BBC News . November 21 , 1999 . Archived from the original on June 21 , 2015 . Retrieved August 1 , 2014 . Jump up ^ Davis , Kristin ( November 2000 ) . ``
I 'm Rich ! ( Now what ? ) '' . Kiplinger 's Personal Finance . pp. 93 -- 94 . ^ Jump up to : Duca , Lauren ( August 15 , 2014 ) . `` The Final Answer On Who
Wants To Be A Millionaire ? , 15 Years After It Premiered '' .
</item 1>

<item 2>
John Carpenter (game show contestant)
John Carpenter ( game show contestant ) - wikipedia John Carpenter ( game show contestant ) Jump to : navigation , search John Carpenter 1968 ( age 48 -- 49 )
Education Rutgers University Occupation IRS agent Known for First top prize winner on Who Wants to Be a Millionaire Home town Northampton , Massachusetts John
Carpenter ( born c. 1968 ) is an American game show contestant and IRS agent . He is best known for becoming the first top prize winner on the United States
version of Who Wants to Be a Millionaire . He held the record for the largest single win in United States game show history , until it was broken by Rahim
Oberholtzer who won $1.12 million on another U.S. quiz show , Twenty One . Carpenter was also the first top prize winner among all international versions of the
Who Wants to Be a Millionaire ? series . On the November 19 , 1999 , episode of Millionaire , Carpenter proceeded to advance to the million - dollar question
without using any lifelines . He then used his Phone - A-Friend to call his father not for help , but rather to tell him he was going to win the game .
</item 2>

<item 3>
John Carpenter (game show contestant)
Carpenter answered the question correctly and became the show 's first millionaire . His win gave him national recognition and led to multiple talk show
appearances , as well as reappearances on Millionaire itself . Contents ( hide ) 1 Career and family 2 Who Wants to Be a Millionaire 3 Later work 3.1 Other game
show appearances 4 See also 5 References 6 External links Career and family ( edit ) Carpenter is from Northampton , Massachusetts . His father , Tom , worked
as a computer program analyst for the Department of Veterans Affairs , while his mother , Gail , served as an administrative assistant for the Massachusetts
Audubon Society . In 1986 , he enrolled at Rutgers University and graduated in 1990 with a degree in economics . In January 1991 , he joined the Internal
Revenue Service after completing government exams and tests . In November 1996 , Carpenter met his future wife , Deborah , who was studying for a master 's
degree at Southern Connecticut State University and worked as a manager of a Fleet Bank branch in New Haven . They married in August 1998 .
</item 3>

<item 4>
John Carpenter, an IRS agent from Northampton, Massachusetts, graduated from Rutgers in 1990. On November 19, 1999, at age 31, he became the first U.S. and
worldwide *Who Wants to Be a Millionaire?* contestant to win the top prize, reaching the $1 million question without using lifelines. He called his father only
to announce he knew the answer. A Champions Edition appearance in 2000 added $250,000, bringing his total
</item 4>

<item 5>
John Carpenter (game show contestant)
They married in August 1998 . At the time of his appearance on Millionaire , he was 31 years old . When he revealed his profession as an IRS officer on
Millionaire , Carpenter was playfully booed by the audience . Who Wants to Be a Millionaire ( edit ) $1 million ( 15 of 15 ) - no time limit Which of these U.S.
Presidents appeared on the television series `` Laugh - In '' ? A : Lyndon Johnson B : Richard Nixon C : Jimmy Carter D: Gerald Ford Carpenter was originally
uninterested in Millionaire but eventually tuned in one night after dinner while having friends over at his house . When he found the show 's higher - tier
questions to not be , in his words , `` any more difficult '' than the lower - tier ones , he decided to call in to the show 's hotline for a chance to become a
contestant . Carpenter answered all of the hotline questions correctly and was on the show within two days . Host Regis Philbin described Carpenter as having ``
cruised right through those first fourteen questions , '' he had proceeded to reach the final question without using any of his lifelines .
</item 5>

<item 6>
John Carpenter (game show contestant)
He explained to Kiplinger 's Personal Finance that `` after the taxes , it 's not change - your - life kind of money if you want to eat every day . '' Carpenter
also described the fame as having a bigger impact on his life than the money , later stating : `` The money does n't change your life . What happens afterwards
might . '' Later work ( edit ) Shortly after winning on Millionaire , Carpenter played himself in a Saturday Night Live skit . Donald Trump , played by Darrell
Hammond , announced that Carpenter would be his running - mate in the presidential election . Afterward , Carpenter pretended to call his father , then shouted
, `` Live from New York , it 's Saturday Night ! '' Carpenter also appeared on Good Morning America , Oprah Winfrey Show , Live with Regis and Kathie Lee , and
Late Show with David Letterman . Carpenter appeared as himself in the second half of the fourth season of Oz . He plays a contestant in a fictional TV game show
called Up Your Ante that the prisoners in Em City are watching .
</item 6>

<item 7>
John Carpenter (game show contestant)
He later participated as part of the Mob on NBC 's 1 vs. 100 on October 27 , 2006 , and as a contestant on the Game Show Network game show Grand Slam .
Carpenter appeared on the August 16 , 2009 episode of Millionaire in prime time for its tenth anniversary . In the audience with him were his father , his wife
, and his son . Additionally , he was also the first expert in the `` Ask the Expert '' lifeline for the eighth season of the syndicated series . See also (
edit ) Biography portal List of Who Wants to Be a Millionaire ? top prize winners References ( edit ) Jump up ^ `` Millionaire Makeovers ! '' . People . Vol. 54
no . 20 . Time Inc . November 13 , 2000 . Archived from the original on September 11 , 2015 . Retrieved July 8 , 2015 . Jump up ^ Petrozello , Donna ( February
3 , 2000 ) . `` Million - plu $ reply an educated guess '' . New York Daily News . p. 101 .
</item 7>

<item 8>
John Carpenter (game show contestant)
The $1 million question was , `` Which of these U.S. Presidents appeared on the television series ' Laugh - In ' ? '' , with the choices being A ) Lyndon
Johnson , B) Richard Nixon , C ) Jimmy Carter , and D ) Gerald Ford . Carpenter used his Phone - A-Friend lifeline to call his father not for help , but rather
to inform him that he knew the answer . Carpenter later explained , `` I thought I 'd look so cocky if I did n't use any lifelines , so I faked it . '' With his
win , Carpenter became the first contestant in the worldwide Millionaire franchise to win the show 's top prize . He said that the only question that had
flustered him was one which asked for the location of the Gunfight at the O.K. Corral . Carpenter eventually remembered that the film Tombstone included the
gunfight , and he replied correctly with the answer ' Tombstone , Arizona ' . While taking a vacation after his win , Carpenter considered quitting his job with
the IRS , but eventually decided against it .
</item 8>

<item 9>
John Carpenter (game show contestant)
The show within the show is hosted by Gordon Elliott , with Eartha Kitt and Didi Conn appearing as celebrity participants . With Rod L. Evans , Carpenter
co-authored a trivia book titled Matching Wits With the Million - Dollar Mind : The World 's Hardest Trivia Quizzes From America 's First Quiz Show Millionaire
. The book was published by Berkley Books in 2002 . Other game show appearances ( edit ) In 2000 , Carpenter appeared in the Who Wants to Be a Millionaire
Champions Edition , in which previous contestants who won $250,000 -- $1,000,000 played again , with half of their additional winnings from the Champions
Edition going to a charity of their choice . Carpenter won $250,000 , bringing his total Millionaire winnings to $1,250,000 . In 2004 , Carpenter participated
in Super Millionaire , as one of the `` Three Wise Men '' on the episode during which Robert `` Bob - O '' Essig won $1,000,000 . He later participated as part
of the Mob on NBC 's 1 vs.
</item 9>

<item 10>
John Carpenter (game show contestant)
`` Miami ' Millionaire ' gets 2nd shot at final answer '' . St. Petersburg Times . Archived from the original on April 5 , 2008 . Retrieved October 9 , 2014 .
Jump up ^ Bracht , Mel ( May 26 , 2000 ) . `` Money is n't the root of happiness : Tulsan has more fun in 2nd Millionaire gig '' . NewsOK . Oklahoma Publishing
Company . Archived from the original on March 13 , 2017 . Retrieved October 9 , 2014 . Jump up ^ Super Millionaire . Season 1 . Episode 2 . February 23 , 2004 .
ABC . Jump up ^ 1 vs. 100 . Season 1 . Episode 3 . October 27 , 2006 . NBC . Jump up ^ Owen , Rob ( August 2 , 2007 ) . `` Dennis Miller rallies around game
show all - stars '' . Pittsburgh Post-Gazette . p . WE - 33 .
</item 10>
=== END ITEM id=09b407f2eefdd9b6 ===
