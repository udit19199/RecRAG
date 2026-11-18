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

=== ITEM id=11dc6df7703564eb ===
Gold sentences and facts:
Sentence 0: German autobahns have no federally mandated speed limit for some classes of vehicles . However , limits are posted ( and enforced ) in areas that are urbanized , substandard , accident - prone , or under construction . On speed - unrestricted stretches , an advisory speed limit ( Richtgeschwindigkeit ) of 130 kilometres per hour ( 81 mph ) applies . While going faster is not illegal as such in the absence of a speed limit , it can cause an increased liability in the case of an accident ; courts have ruled that an `` ideal driver '' who is exempt from absolute liability for `` inevitable '' tort under the law would not exceed Richtgeschwindigkeit .
  Fact 0.0: German autobahns have no federally mandated speed limit for some classes of vehicles.
  Fact 0.1: Speed limits are posted in areas that are urbanized.
  Fact 0.2: Speed limits are posted in areas that are substandard.
  Fact 0.3: Speed limits are posted in areas that are accident-prone.
  Fact 0.4: Speed limits are posted in areas that are under construction.
  Fact 0.5: Posted speed limits are enforced.
  Fact 0.6: On speed-unrestricted stretches, an advisory speed limit applies.
  Fact 0.7: The advisory speed limit is called Richtgeschwindigkeit.
  Fact 0.8: The advisory speed limit is 130 kilometres per hour (81 mph).
  Fact 0.9: Going faster is not illegal as such in the absence of a speed limit.
  Fact 0.10: Going faster can cause an increased liability in the case of an accident.
  Fact 0.11: Courts have ruled that an 'ideal driver' is exempt from absolute liability for 'inevitable' tort under the law.
  Fact 0.12: Courts have ruled that an 'ideal driver' would not exceed Richtgeschwindigkeit.

Retrieved items:
<item 1>
Autobahn
These limiters can be deactivated , so speeds up to 300 km / h ( 190 mph ) might arise on the German autobahn , but due to other traffic , such speeds are
generally not attainable except during certain times like between 10 p.m. and 6 a.m. or on Sundays ( when trucks drivers have to rest by law ) . Furthermore ,
there are certain autobahn sections which are known for having light traffic , making such speeds attainable during most days ( especially some of those located
in Eastern Germany ) . Most unlimited sections of the autobahn are located outside densely populated areas . Vehicles with a top speed less than 60 km / h ( 37
mph ) ( such as quads , low - end microcars , and agricultural / construction equipment ) are not allowed to use the autobahn , nor are motorcycles and scooters
with low engine capacity regardless of top speed ( mainly applicable to mopeds which are typically limited to 25 or 45 km / h anyway ) . To comply with this
limit , heavy - duty trucks in Germany ( e.g. mobile cranes , tank transporters etc . )
</item 1>

<item 2>
Autobahn
The official German term is Bundesautobahn ( plural Bundesautobahnen , abbreviated BAB ) , which translates as `` federal motorway '' . The literal meaning of
the word Bundesautobahn is `` Federal Auto ( mobile ) Track '' . German autobahns have no federally mandated speed limit for some classes of vehicles . However
, limits are posted ( and enforced ) in areas that are urbanized , substandard , accident - prone , or under construction . On speed - unrestricted stretches ,
an advisory speed limit ( Richtgeschwindigkeit ) of 130 kilometres per hour ( 81 mph ) applies . While going faster is not illegal as such in the absence of a
speed limit , it can cause an increased liability in the case of an accident ; courts have ruled that an `` ideal driver '' who is exempt from absolute
liability for `` inevitable '' tort under the law would not exceed Richtgeschwindigkeit . A 2008 estimate reported that 52 % of the autobahn network had only
the advisory speed limit , 15 % had temporary speed limits due to weather or traffic conditions , and 33 % had permanent speed limits .
</item 2>

<item 3>
Autobahn
Some speed limits are implemented on different autobahns . Certain limits are imposed on some classes of vehicles : 60 km / h ( 37 mph ) Buses carrying standing
passengers Motorcycles pulling trailers 80 km / h ( 50 mph ) Vehicles with maximum allowed weight exceeding 3.5 t ( except passenger cars ) Passenger cars and
trucks with trailers Buses 100 km / h ( 62 mph ) Passenger cars pulling trailers certified for 100 km / h Buses certified for 100 km / h not towing trailers
Additionally , speed limits are posted at most on - and off - ramps and interchanges and other danger points like sections under construction or in need of
repair . Where no general limit is required , the advisory speed limit is 130 km / h ( 81 mph ) , referred to in German as the Richtgeschwindigkeit . The
advisory speed is not enforceable ; however , being involved in an accident driving at higher speeds can lead to the driver being deemed at least partially
responsible due to `` increased operating danger '' ( Erhöhte Betriebsgefahr ) .
</item 3>

<item 4>
Autobahn
Certain stretches have lower speed limits during wet weather . Some areas have a speed limit of 120 km / h ( 75 mph ) in order to reduce noise pollution during
overnight hours ( usually 10pm -- 6am ) or because of increased traffic during daytime ( 6am -- 8pm ) . Dynamic traffic signs on an Autobahn Some limits were
imposed to reduce pollution and noise . Limits can also be temporarily put into place through dynamic traffic guidance systems that display the according
message . More than half of the total length of the German autobahn network has no speed limit , about one third has a permanent limit , and the remaining parts
have a temporary or conditional limit . Some cars with very powerful engines can reach speeds of well over 300 km / h ( 190 mph ) . Major German car
manufacturers , except Porsche , follow a gentlemen 's agreement by electronically limiting the top speeds of their cars -- with the exception of some top of
the range models or engines -- to 250 km / h ( 155 mph ) .
</item 4>

<item 5>
Speed limit
Feb 2008 . Retrieved 2010 - 10 - 16 . Currently , 52 % of the German motorways do not have a speed limit , 15 % have temporary speed limits due to weather or
traffic conditions and 33 % have permanent speed limits . On unlimited sections there is a 130 km / h recommendation . Jump up ^ `` § 3 StVO : Speed '' .
Bundesministerium der Justiz und für Verbraucherschutz . Retrieved 2014 - 04 - 29 . Diese Geschwindigkeitsbeschränkung gilt nicht ( ... ) auf anderen Straßen
mit Fahrbahnen für eine Richtung , die durch Mittelstreifen oder sonstige bauliche Einrichtungen getrennt sind . Sie gilt ferner nicht auf Straßen , die
mindestens zwei durch Fahrstreifenbegrenzung ( Zeichen 295 ) oder durch Leitlinien ( Zeichen 340 ) markierte Fahrstreifen für jede Richtung haben .
</item 5>

<item 6>
Speed limit
This speed zone is scheduled to be activated November 2016 . Roads without speed limits ( edit ) German border crossing sign showing 50 km / h ( 31 mph ) limit
in built - up areas , 100 km / h ( 62 mph ) in rural areas , but a legal advisory 130 km / h ( 81 mph ) limit for the Autobahn ( motorway ) , increasing
liability in the case of an accident from driving faster Just over half of the German autobahns have only an advisory speed limit ( called in German
Richtgeschwindigkeit ) , 15 % have temporary speed limits due to weather or traffic conditions and 33 % have permanent speed limits , according to 2008
estimates . German federal highways and any road outside of towns which is either a dual carriageway or features at least two lanes per direction do have a
general speed limit of 100 km / h . Usually it is reduced to 80 km / h at Allée - streets ( federal highways bordered by trees or bushes on one or both sites )
.
</item 6>

<item 7>
Autobahn
mobile cranes , tank transporters etc . ) often have a maximum design speed of 62 km / h ( 39 mph ) ( usually denoted by a round black - on - white sign with ``
62 '' on it ) , along with flashing orange beacons to warn approaching cars that they are travelling slowly . There is no general minimum speed but drivers are
not allowed to drive at an unnecessarily low speed as this would lead to significant traffic disturbance and an increased collision risk . Public debate ( edit
) German national speed limits have a historical association with war - time restrictions and deprivations , the Nazi era , and the Soviet era in East Germany .
`` Free driving for free citizens '' ( `` freie Fahrt für freie Bürger '' ) , a slogan promoted by the German Auto Club since the 1970s , is a popular slogan
among those opposing autobahn speed restrictions .
</item 7>

<item 8>
Autobahn
In 2006 , speeds were recorded using automated detection loops in the State of Brandenburg at two points : on a six - lane section of A9 near Niemegk with a 130
km / h ( 81 mph ) advisory speed limit ; and on a four - lane section of A10 bypassing Berlin near Groß Kreutz with a 120 km / h ( 75 mph ) mandatory limit .
</item 8>

<item 9>
Autobahn
Truck stop Scandinavian Park off the A 7 Rest area Dammer Berge on the A 1 Rest areas and truck stops are marked several times , starting several kilometres in
advance , and with larger signs that often include icons announcing what kinds of facilities travellers can expect . Speed limits ( edit ) Autobahn with three
separate lanes in each direction and an emergency lane On the autobahns there is an advisory speed limit ( Richtgeschwindigkeit ) of 130 km / h ( unless
otherwise regulated by signs ) . The `` limits no longer apply '' ( Ende aller Streckenverbote ) sign , indicating a return to the default speed , while lifting
all other limits as well ( all limits are indicated by round signs with red border ) . GPS tracks colored according to speed show considerable speed differences
at an autobahn crossing Germany 's autobahns are famous for being among the few public roads in the world without blanket speed limits for cars and motorcycles
. As such , they are important German cultural identifiers , `` ... often mentioned in hushed , reverential tones by motoring enthusiasts and looked at with a
mix of awe and terror by outsiders . '' Some speed limits are implemented on different autobahns .
</item 9>

<item 10>
Autobahn
The Federal Road Research Institute ( Bundesanstalt für Straßenwesen ) solicited information about speed regulations on autobahns from the sixteen States and
reported the following , comparing the years 2006 and 2008 : Parameter 2006 2008 Change Autobahn Carriageway - Km 24,735 km 25,240 km + 505 km Advisory Limit
Only 69.2 % 65.5 % - 580 km Variable Limit ( with Advisory Maximum ) 4.2 % 4.1 % - 5 km Permanent or Conditional Speed Limit 26.7 % 30.4 % + 1,090 km Except at
construction sites , the general speed limits , where they apply , are usually between 100 km / h ( 62 mph ) and 130 km / h ( 81 mph ) ; construction sites
usually have a speed limit of 80 km / h ( 50 mph ) but the limit may be as low as 60 km / h ( 37 mph ) . In rare cases , sections may have limits of 40 km / h (
25 mph ) , or on one ramp 30 km / h ( 19 mph ) .
</item 10>
=== END ITEM id=11dc6df7703564eb ===

=== ITEM id=ae987810e039cbd6 ===
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
Stanford , California Championship Saturday , May 9 , 2015 Maples Pavilion , Stanford , California Lewis ( 3 ) 25 22 25 25 Hawaii ( 1 ) 22 20 25 25 5 Penn State
( 1 ) 20 25 16 20 5 Penn State ( 3 ) 25 25 17 27 Lewis ( 2 ) 25 23 15 27 21 Loyola ( Chicago ) ( 3 ) 21 25 25 25 23 UC Irvine ( 0 ) 22 19 17 Loyola ( Chicago )
( 3 ) 25 33 25 Loyola ( Chicago ) ( 3 ) 25 25 25 6 Pfeiffer ( 0 ) 20 31 15 2016 ( edit ) Main article : 2016 NCAA Men 's National Collegiate Volleyball
Tournament See also ( edit ) NCAA Division III Men 's Volleyball Championship NCAA Women 's Volleyball Championship List of NCAA men 's volleyball schools
Pre-NCAA Intercollegiate Volleyball Champions Notes ( edit ) Jump up ^ Athletic Business ,
</item 1>

<item 2>
NCAA Men's National Collegiate Volleyball Championship
Champions ( edit ) NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner
Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2
-- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving
Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State (
21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara ,
</item 2>

<item 3>
NCAA Men's National Collegiate Volleyball Championship
-- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships , -- 1 championship School # Year won UCLA 19 1970 , 1971 , 1972 , 1974 , 1975
, 1976 , 1979 , 1981 , 1982 , 1983 , 1984 , 1987 , 1989 , 1993 , 1995 , 1996 , 1998 , 2000 , 2006 Pepperdine 5 1978 , 1985 , 1986 , 1992 , 2005 Southern
California 1977 , 1980 , 1988 , 1990 UC Irvine 2007 , 2009 , 2012 , 2013 Ohio State 2011 , 2016 , 2017 BYU 1999 , 2001 , 2004 Penn State 1994 , 2008 Stanford
1997 , 2010 Loyola Chicago 2014 , 2015 Vacated 2002 , 2003 Long Beach State 1991 San Diego State 1973 Past tournaments ( edit ) Historically ,
</item 3>

<item 4>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details
Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018
Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George
Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion † Vacated due to NCAA
violations Team titles ( edit ) UCLA Pepperdine USC UC Irvine BYU Penn State Stanford Loyola Ohio State ⇙ Hawaii Lewis Long Beach State San Diego State Schools
with national championships -- 19 championships , -- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships ,
</item 4>

<item 5>
NCAA Men's National Collegiate Volleyball Championship
Gender Equity -- Boys ' and Mens ' Volleyball Participation Continues to Lag , April 2009 Jump up ^
http://fs.ncaa.org/Docs/stats/m_volleyball_champs_records/2012/d1/2012d1mvbchamp.pdf Jump up ^ UC Irvine defeats BYU in three sets to claim consecutive national
title , NCAA.com , May 5 , 2013 Jump up ^ Loyola repeats as men 's NCAA volleyball national champions , Chicago Tribune , May 10 , 2015 Jump up ^ NCAA Tourney
to have 2 play - in matches starting in 2014 , Off the Block , October 23 , 2013 NCAA Men 's National Collegiate Volleyball Championship Tournaments 1970 1971
1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006
</item 5>

<item 6>
NCAA Men's National Collegiate Volleyball Championship
May 2 , 2013 Pauley Pavilion , Los Angeles Championship Saturday , May 4 , 2013 Pauley Pavilion , Los Angeles Brigham Young ( 3 ) 25 25 25 Penn State ( 0 ) 21
16 22 Brigham Young ( 0 ) 23 22 24 UC Irvine ( 3 ) 25 25 26 UC Irvine ( 3 ) 26 25 29 Loyola University Chicago ( 0 ) 24 18 27 2014 ( edit ) The semifinals and
finals 2014 tournament were held in the Gentile Arena in Chicago on the campus of Loyola University Chicago . Two quarterfinal `` play - in '' matches were held
at the Gentile Arena two days prior to the national semifinals , as the 2014 tournament expanded to six teams for the first time ever . A second at - large was
added to the field , and the champions of the newly - eligible Conference Carolinas men 's volleyball division got an automatic qualification .
</item 6>

<item 7>
NCAA Men's National Collegiate Volleyball Championship
California - based universities have dominated the men 's volleyball national championship ; Loyola Chicago , Penn State , Ohio State and BYU are the only
non-California universities to have won the National Collegiate championship ; Hawaii and Lewis also won the championship tournament , but had their victories
vacated due to NCAA rules violations . Only seven non-California universities have participated in the National Collegiate championship match ( Loyola , BYU ,
Penn State , Ohio State , IPFW , Hawaii , and Lewis ) , although other universities such as Princeton and Ball State have participated in the final four . Only
three finals have involved two non-California schools : the 2003 final , when Lewis defeated BYU but had its win vacated ; the 2015 final , in which Loyola
defeated Lewis ; and the 2016 final , when Ohio State defeated BYU . UCLA , Southern California , Penn State , Stanford , and Long Beach State are the only
schools in Division I to have won an NCAA national championship in both men and women 's volleyball .
</item 7>

<item 8>
NCAA Men's National Collegiate Volleyball Championship
( But for Hawaii 's vacated 2002 title , it would have matched this feat with 3 NCAA and 1 AIAW women 's national championships . ) In addition , Stanford (
1996 -- 97 ) and Penn State ( 2007 -- 08 ) are the only universities whose men and women 's volleyball programs won the national championship in the same
academic year . The 2016 championship tournament took place from May 3 -- 7 at Rec Hall on the campus of Penn State . 2011 ( edit ) May 5 , 2011 -- UC Santa
Barbara def . Southern California , 29 - 27 , 24 - 26 , 25 - 15 , 25 - 18 ; Ohio State def . Penn St. , 25 - 18 , 24 - 26 , 25 - 22 , 25 - 23 May 7 , 2011 --
Ohio State def .
</item 8>

<item 9>
NCAA Men's National Collegiate Volleyball Championship
California ( Long Beach State ) The Pyramid BYU ( 23 -- 4 ) 3 -- 0 UCLA Ohio State and Penn State 2002 Details University Park , Pennsylvania ( Penn State ) Rec
Hall Hawaii ( 24 -- 8 ) † 3 -- 1 Pepperdine Ball State and Penn State 2003 Details Long Beach , California ( Long Beach State ) The Pyramid Lewis ( 29 -- 6 ) †
3 -- 2 BYU Penn State and Pepperdine Details Honolulu , HI ( Hawaii ) Stan Sheriff Center BYU ( 29 -- 4 ) 3 -- 2 Long Beach State Lewis and Penn State 2005
Details Los Angeles ( UCLA ) Pauley Pavilion Pepperdine ( 25 -- 2 ) 3 -- 2 UCLA Ohio State and Penn State 2006 Details University Park , Pennsylvania ( Penn
State ) Rec Hall UCLA ( 26 -- 12 ) 3 -- 0 Penn State UC Irvine and IPFW 2007 Details Columbus , Ohio ( Ohio State ) St. John Arena UC Irvine ( 29 -- 5 ) 3 -- 1
IPFW Penn State and Pepperdine 2008 Details Irvine ,
</item 9>

<item 10>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen
Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26
-- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and
Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC
Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park ,
</item 10>
=== END ITEM id=ae987810e039cbd6 ===

=== ITEM id=1a8114555aa4e898 ===
Gold sentences and facts:
Sentence 0: As a young woman , Rogue permanently absorbed the superhuman powers and the psyche of Carol Danvers , the original Ms. Marvel . This provided her with superhuman strength , stamina , durability , reflexes , speed and a 7th sense . She was able to repel bullets and fly at sub-sonic speeds , much like Ms. Marvel could . In addition , she possessed an amalgamated mutant human / Kree physiology that rendered her resistant to most toxins and poisons , with the added effect of making her virtually invulnerable .
  Fact 0.0: Rogue, as a young woman, permanently absorbed the superhuman powers of Carol Danvers.
  Fact 0.1: Rogue, as a young woman, permanently absorbed the psyche of Carol Danvers.
  Fact 0.2: Carol Danvers is the original Ms. Marvel.
  Fact 0.3: Rogue's absorption gave Rogue superhuman strength.
  Fact 0.4: Rogue's absorption gave Rogue superhuman stamina.
  Fact 0.5: Rogue's absorption gave Rogue superhuman durability.
  Fact 0.6: Rogue's absorption gave Rogue superhuman reflexes.
  Fact 0.7: Rogue's absorption gave Rogue superhuman speed.
  Fact 0.8: Rogue's absorption gave Rogue a 7th sense.
  Fact 0.9: Rogue was able to repel bullets.
  Fact 0.10: Rogue was able to fly at sub-sonic speeds.
  Fact 0.11: Rogue's ability to fly at sub-sonic speeds was much like Ms. Marvel's.
  Fact 0.12: Rogue possessed an amalgamated mutant human/Kree physiology.
  Fact 0.13: Rogue's amalgamated mutant human/Kree physiology made Rogue resistant to most toxins.
  Fact 0.14: Rogue's amalgamated mutant human/Kree physiology made Rogue resistant to most poisons.
  Fact 0.15: The added effect of Rogue's amalgamated physiology was that Rogue was virtually invulnerable.

Retrieved items:
<item 1>
Rogue (comics)
Z'Cann uses her telepathy to amplify Rogue 's capacity to assimilate memories , causing her powers to mutate . Henceforth , Rogue is able to `` recall ''
previously absorbed powers , but also finds it increasingly difficult to control which powers manifest when , typically possessing Wolverine 's healing factor
and claws , and Cyclops ' optic blasts along with her usual strength . X-Treme X-Men team ( edit ) Rogue was part of the X-Treme X-Men team led by Storm . The
team 's first mission was in search of Destiny 's Diaries ( which prophesied future events ) . During an invasion of Khan ( an alien conqueror from another
dimension ) of the island nation of Madripoor , Rogue requested that Sage use her power of jump - starting abilities to evolve Rogue to a point where she could
control all of the various powers that she has ever imprinted . Sage agreed , and Rogue became a one - woman army , able to use the powers of anyone she had
absorbed in the past all at once . During Khan 's invasion , Rogue was also confronted by Vargas who was said to be `` a new species altogether '' .
</item 1>

<item 2>
Rogue (comics)
She possesses superhuman strength sufficient enough to lift well over 100 tons , as well as some degree of superhuman speed , agility , and reflexes . In
addition , Rogue is virtually invulnerable to damage and is capable of flying at high speeds . Rogue 's eyes glow in darkness and her vision extends somewhat
into the infrared spectrum , allowing her to see in the dark . Her new ionic physiology also grants her a form of immortality , protecting her from age and
disease . Natural skills ( edit ) Rogue has shown a fair knowledge of unarmed combat when in a position where she has to fight in situations where she can not
simply use her powers to knock her foe out . She has also received a degree of training from Professor X to protect herself against even subtle telepathic
attacks , this training allowing her to escape Kobik 's reprogramming of various Avengers and villains to reassemble her teammates against this new threat when
even experienced telepaths like Cable were trapped in the illusion . Rogue 's real name ( edit ) In the Going Rogue arc of her 2004 ongoing series , Rogue 's
real name is finally established as Anna Marie .
</item 2>

<item 3>
Rogue (comics)
Powers and abilities ( edit ) This section relies too much on references to primary sources . Please improve this section by adding secondary or tertiary
sources . ( March 2015 ) ( Learn how and when to remove this template message ) This section describes a work or element of fiction in a primarily in - universe
style . Please help rewrite it to explain the fiction more clearly and provide non-fictional perspective . ( March 2015 ) ( Learn how and when to remove this
template message ) Due to the nature of her unique mutant powers , Rogue has had several different abilities over the decades . Mutant powers ( edit ) Cover art
of Rogue vol. 3 , # 3 ( September 2004 ) ; art by Rodolfo Migliari . Rogue possesses the mutant ability to absorb the psyche and abilities of another human
being ( or members of some sapient alien races ) through skin contact . Rogue can absorb the memories , knowledge , talents , personality and physical abilities
( whether superhuman or not ) of the person she touches , as well as occasionally duplicating in herself physical characteristics of her victim .
</item 3>

<item 4>
Rogue (comics)
Rogue 's victim can either resist the absorption and suffer , like originally with Rogue 's powers , or instead submit to the absorption , and so the process
instead becomes entirely fluent without consequences . Rogue has used this twist extensively in the Legacy series ; just two examples of it are when aiding the
victims of a subway collapse absorbing all rescuers ' abilities in X-Men Legacy 274 , and also when quenching a riot in a prison of mutants absorbing the
existence of some mutant - volunteers in X-Men Legacy 275 . During Uncanny Avengers , Rogue used her powers to absorb the energies of numerous superheroes to
oppose the Celestial Executioner , stating that she 'd `` been practicing '' ; she was even able to absorb the ionic energy based powers of Wonder Man ( whom
she 'd been previously unable to absorb ) . However , apparently due to the scale of the energies she absorbed in the process , Rogue 's powers have reverted to
their original state , unable to touch anyone without absorbing them automatically , and she retains Wonder Man 's psyche and powers . Ms.
</item 4>

<item 5>
Rogue (comics)
Over time , Rogue 's own abilities returned , although exactly how and when was never revealed or discussed . However , she no longer possessed the abilities
she had absorbed from Ms. Marvel . Rogue and Gambit were both put on Havok 's team and participated in various missions . As described in her own miniseries ,
Rogue then traveled down south to save a young mutant girl from her powers . While there , she met Campbell St. Ange , a young man who was immune to Rogue 's
lethal touch . Also while there , Rogue forcibly absorbed knowledge from her Aunt Carrie that explained that Rogue 's mother had traveled to the Far Banks , a
dream - realm , to stop her father from getting there himself . Rogue encountered the incorporeal spirit of her mother therein , and absorbed her memories .
After the reunion , her mother 's trapped spirit could finally move on . Rogue subsequently went back to her Aunt Carrie and made amends with her .
</item 5>

<item 6>
Rogue (comics)
She also still possessed a `` fluid genome '' that enabled Sage to use her as a conduit through which to channel the mutant powers of the X-Treme X-Men team in
a fight against Bogan . Sunfire powers ( edit ) In the short - lived volume three of her self - entitield series ( 2004 -- 2005 ) , Rogue absorbed a large
portion of the mutant Sunfire 's solar - absorption based powers . In addition to her own natural mutant abilities Rogue could then project intense heat and
flame , envelop her body in a fiery aura , fly by focusing her power downwards in a tight stream to propel her like a rocket , focus her power inward to
increase her strength ( though not at her Ms. Marvel levels ) , exercise immunity to heat and radiation , and see the infra - red spectrum . Rogue was
eventually purged of these abilities when she came into contact with Hope Summers . Wonder Man powers ( edit ) Rogue currently possesses Wonder Man 's essence ,
including his powers and psyche . Her organic tissues have been permeated with ionic energy , granting her an array of superhuman abilities .
</item 6>

<item 7>
Rogue (comics)
Ms. Marvel powers ( edit ) As a young woman , Rogue permanently absorbed the superhuman powers and the psyche of Carol Danvers , the original Ms. Marvel . This
provided her with superhuman strength , stamina , durability , reflexes , speed and a 7th sense . She was able to repel bullets and fly at sub-sonic speeds ,
much like Ms. Marvel could . In addition , she possessed an amalgamated mutant human / Kree physiology that rendered her resistant to most toxins and poisons ,
with the added effect of making her virtually invulnerable . Rogue on the cover of Rogue vol 3 # 7 ( March 2005 ) ; art by Frank G. D'Armata , Scot Eaton , and
Don Hillsman Rogue also gained a precognitive `` seventh sense '' that enables her to predict an enemy 's move subconsciously during battle . She used this
ability to predict where Nightcrawler would teleport and from which direction Magus of the Technarchy would attack . This ability was not always reliable ,
however , and would randomly and sporadically activate .
</item 7>

<item 8>
Rogue (comics)
When Professor Xavier removes the blocks , her powers are allowed to develop normally . With this new control , Rogue demonstrates the ability to absorb and
collectively utilize all the powers of the New X-Men , in order to defeat a rogue Predator X , with no apparent harm to either them or herself . Her power also
affects aliens , such as when she was transferred to a planet in another dimension by Magik , and one of the aliens involuntarily touches her and she gains the
ability to communicate in their language , along with their knowledge . She later uses a more lethal version of her powers against the Avengers during her fight
with them , even making Falcon and She - Hulk immediately unconscious without any negative feedback on her physical self ( besides She - Hulk 's green skin )
but she can not control the minds of the ones she absorbs . She absorbs their powers and renders them unconscious for a long time . The new twist in Rogue 's
powers consists of prompting her victims of her skin - to - skin contact , for a consent on the process .
</item 8>

<item 9>
Graph query row:
rogue: Rogue
relationship: RELATED_TO
fact: Rogue permanently absorbed Carol Danvers's powers and psyche.
context: None
relationship_detail: None
relation: bumps into
</item 9>

<item 10>
Rogue (comics)
Through meditation , Rogue learned how to suppress the random manifestations of most of these abilities , but found herself with a new ' default ' state for a
time , where she possessed Wolverine 's claws and healing factor , Cyclops ' optic blasts - requiring her to wear ruby quartz glasses - while also still
retaining her mutant and Ms. Marvel powers . Later , Sage used her own mutant abilities to ' jump - start ' Rogue 's power , giving her the control to manifest
any ability she had ever absorbed , whenever she wished to . It is generally accepted that after regaining her abilities after X-Treme X-Men , she has lost this
ability . Though Z'Cann was a telepath , Rogue never displayed this power ( though she did manifest telepathy , it was with the Phoenix energy form , so it was
most likely an echo of Jean Grey 's powers ) . Powerless ( edit ) For a time Rogue lost all of her powers , including her original mutant ones , after she was
fatally injured in a fight with the mysterious warrior Vargas . During this period , she displayed exceptional fighting skills and agility , though it was said
these were not superhuman in nature .
</item 10>
=== END ITEM id=1a8114555aa4e898 ===

=== ITEM id=760caed385dc2135 ===
Gold sentences and facts:
Sentence 0: With the Intertropical Convergence Zone ( ITCZ ) swinging northward over West Africa from the Southern Hemisphere in April , heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical ( south westerlies ) and the Continental tropical ( north easterlies ) , begins in central Nigeria while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity , heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in central Nigeria varies from 1,100 mm ( 43.3 in ) in the lowlands of the river Niger Benue trough to over 2,000 mm ( 78.7 in ) along the south western escarpment of the Jos Plateau .
  Fact 0.0: The Intertropical Convergence Zone (ITCZ) swings northward over West Africa from the Southern Hemisphere in April.
  Fact 0.1: In April, heavy showers come from pre-monsoonal convective clouds, mainly in the form of squall lines.
  Fact 0.2: The squall lines are also known as the north easterlies.
  Fact 0.3: The north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria.
  Fact 0.4: The two dominant airmasses in Nigeria are the Maritime tropical (south westerlies) and the Continental tropical (north easterlies).
  Fact 0.5: The pre-monsoonal showers begin in central Nigeria.
  Fact 0.6: The monsoons from the South Atlantic Ocean arrive in central Nigeria in July.
  Fact 0.7: The monsoons bring high humidity, heavy cloud cover and heavy rainfall to central Nigeria in July.
  Fact 0.8: The rainfall from the monsoons can be a daily occurrence in central Nigeria.
  Fact 0.9: The monsoon rainfall in central Nigeria lasts until September.
  Fact 0.10: The monsoons gradually begin retreating southward to the southern part of Nigeria when September arrives.
  Fact 0.11: Rainfall totals in central Nigeria vary from 1,100 mm (43.3 in) in the lowlands of the river Niger Benue trough.
  Fact 0.12: Rainfall totals in central Nigeria reach over 2,000 mm (78.7 in) along the south western escarpment of the Jos Plateau.

Retrieved items:
<item 1>
Geography of Nigeria
Temperatures in the northern part of Nigeria can go as high as 48 ° C ( 118.4 ° F ) in cities like Maiduguri . The high temperatures coupled with an increase in
insolation causes a region of low pressure to develop over west Africa and Nigeria ( between March to May ) . The Tropical continental airmass ( CT ) from the
Sahara Desert in the northern part of West Africa , is weakened due to the overheating of the land surface in west Africa and Nigeria at this time . The
Tropical continental airmass ( CT ) begins to retreat northwards to the Sahara Desert due to massive heating of the land which transfers heat in the form of
convection into the Tropical continental airmass ( CT ) which constitutes the main layer of air above the land . This transfer of heat in the Tropical
continental airmass ( CT ) in turn , causes the wind to expand and become lighter as this is the normal behaviour for winds moving above intensely heated
grounds .
</item 1>

<item 2>
Geography of Nigeria
The northward retreat of the tropical continental airmass ( CT ) , is caused by the sun 's northward shift from the tropic of capricorn in the southern
hemisphere to the tropic of cancer in the northern hemisphere . This shift begins from February and ends in June , when the sun is fully overhead , at the
tropic of cancer in the northern hemisphere . During this northward migration of the sun as a result of the earth tilting along its axis , the sun crosses the
equator ( around March ) , moving over west Africa at this time on its journey to the northern hemisphere . West Africa comes directly under the sun at this
time . The sun is overhead throughout west Africa and over Nigeria during this period of the sun 's northward migration to the tropic of cancer in the northern
hemisphere . The whole of West Africa is heated intensely as result of the increased insolation received from the sun being overhead over west Africa .
Temperatures can climb as high as 35 ° C ( 95.0 ° F ) over west Africa during this time .
</item 2>

<item 3>
Geography of Nigeria
The Tropical continental airmass ( CT ) loses its strength as a major airmass in the region of west Africa and over Nigeria at this time ( around February in
the southern part of Nigeria to June in northern Nigeria ) and begins to retreat coupled with the rising of air in form of convection within this airmass (
Tropical continental airmass ( CT ) ) , further weakening the dominance of the wind over west Africa and Nigeria . The Tropical continental airmass ( CT )
finally retreats from most part of Nigeria , and the West African atmosphere around April to May , leaving an empty atmosphere over Nigeria . The sun 's rays
enters into the atmosphere of Nigeria more intense than it does during the presence of the Tropical continental airmass , which contained dust ( in form of haze
) that reduced the intensity of the sun . The overheating of the west Africa land mass and Nigeria in particular creates a low pressure region over west Africa
and Nigeria . This low pressure zone attracts the Tropical Maritime Airmass ( MT ) from the south Atlantic Ocean since areas of low pressures experiences inward
blowing winds because winds are moving air blowing outwards from regions of high pressure to regions of low pressure .
</item 3>

<item 4>
Geography of Nigeria
Nigeria 's northern end is south of the 15 degrees line at about 14 degrees . Nigeria 's location in the wetter part of the easterly waves south of the 15
degree line creates wetter climatic conditions for Nigeria especially during the monsoons . Tropical continental airmass ( edit ) The Tropical Continental
Airmass ( CT ) locally known as the harmattan , is a wind originating from North Africa which crosses the Sahara Desert into west Africa to Nigeria . This
airmass dominates Nigeria 's climate during the dry season from December to March . The Tropical continental airmass is dusty and creates a haze within the
atmosphere of west Africa and Nigeria when it predominates . The haze is as a result of the dust within the airmass limiting visibility and blocking much of the
sun 's rays from reaching the earth . Effects of the tropical continental airmass ( edit ) The mass has no ability of forming rain due to low humidity within
the airmass , since it crosses the Sahara Desert , it picks up dust instead of water thereby creating little chances for rainfall .
</item 4>

<item 5>
Geography of Nigeria
This climate , the tropical savanna climate exhibits a well marked rainy season and a dry season with a single peak known as the summer maximum due to its
distance from the equator . Temperatures are above 18 ° C ( 64 ° F ) throughout the year . Abuja , Nigeria 's capital city found in central Nigeria , has a
temperature range of 18.45 ° C ( 65.21 ° F ) to 36.9 ° C ( 98.4 ° F ) , and an annual rainfall of about 1,500 mm ( 59.1 in ) with a single rainfall maxima in
September . The single Dry season experienced in this climate , the tropical savanna climate in central Nigeria beginning from December to march , is hot and
dry with the Harmattan wind , a continental tropical ( CT ) airmass laden with dust from the Sahara Desert prevailing throughout this period .
</item 5>

<item 6>
Geography of Nigeria
These are the dry season and the rainy season . The dry season is accompanied by a dust laden airmass from the Sahara Desert , locally known as Harmattan , or
by its main name , The Tropical Continental ( CT ) airmass , while the rainy season is heavily influenced by an airmass originating from the south atlantic
ocean , locally known as the south west wind , or by its main name , The Tropical Maritime ( MT ) airmass . These two major wind systems in Nigeria are known as
the trade winds . Trade winds ( edit ) Tropical Maritime airmass ( edit ) The tropical maritime airmass ( MT ) is responsible for Nigeria 's rainy season . This
wind ( the tropical maritime airmass ) invades the country from February in the southern part of Nigeria while it takes longer for the wind to fully cover the
whole of the country , reaching the northern part of Nigeria in June . Its invasion is as a result of the northward retreat , of the tropical continental
airmass ( CT ) known as the harmattan .
</item 6>

<item 7>
Geography of Nigeria
With the Intertropical Convergence Zone ( ITCZ ) swinging northward over West Africa from the Southern Hemisphere in April , heavy showers coming from
pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two
dominant airmasses in Nigeria known as the Maritime tropical ( south westerlies ) and the Continental tropical ( north easterlies ) , begins in central Nigeria
while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity , heavy cloud cover and heavy rainfall which
can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in
central Nigeria varies from 1,100 mm ( 43.3 in ) in the lowlands of the river Niger Benue trough to over 2,000 mm ( 78.7 in ) along the south western escarpment
of the Jos Plateau . The Sahel climate or tropical dry climate , is the predominant climate type in the northern part of Nigeria . Annual rainfall totals are
lower compared to the southern and central part of Nigeria .
</item 7>

<item 8>
Geography of Nigeria
The Tropical monsoon climate has a very small temperature range . Then temperature ranges are almost constant throughout the year , for example , Warri town in
the southern part of Nigeria , records a maximum of 28 ° C ( 82.4 ° F ) for its hottest month while its lowest temperature is 26 ° C ( 78.8 ° F ) in its coldest
month . The temperature difference of Warri town is not more than 2 ° C ( 5 ° F ) . The southern part of Nigeria experiences heavy and abundant rainfall . These
storms are usually convectional in nature due to the regions proximity , to the equatorial belt . The annual rainfall received in this region is very high ,
usually above the 2,000 mm ( 78.7 in ) rainfall totals giving for tropical rainforest climates worldwide . Over 4,000 mm ( 157.5 in ) of rainfall is received in
the coastal region of Nigeria around the Niger delta area . Bonny town found in the coastal region of the Niger delta area in southern Nigeria receives well
over 4,000 mm ( 157.5 in ) of rainfall annually .
</item 8>

<item 9>
Geography of Nigeria
In the rainy season it is damp , and the rainfalls are usually abundant . The dry season of Nigeria is a period of little cloud cover in the southern part of
Nigeria to virtually no cloud cover in the northern part of Nigeria . The sun shines through the atmosphere with little obstructions from the clear skies making
the dry season in Nigeria a period of warm weather conditions.In the middle of the dry season around December , a dusty wind from the Sahara Desert called the
harmattan enters Nigeria from the northeastern part of the country blocking sun rays partially from shining and also creating haze in the atmosphere , this
activities of the wind lowers temperatures considerably saving inhabitants for sometime , from the scorching heat that would have occurred as a result of
clearer skies during the dry season . But with the withdrawal of this wind around March to April following the onset of the rainy season , temperatures can go
as high as 44 ° C ( 111.2 ° F ) in some parts of Nigeria . Semi temperate weather conditions prevail on the highlands in central Nigeria above 1,200 metres (
3,937 ft ) above sea level , namely the Jos Plateau .
</item 9>

<item 10>
Geography of Nigeria
The rest of the southeast receives between 2,000 and 3,000 mm ( 118.1 in ) of rain per year . The southern region of Nigeria experiences a double rainfall
maxima characterised by two high rainfall peaks , with a short dry season and a longer dry season falling between and after each peaks . The first rainy season
begins around March and last to the end of July with a peak in June , this rainy season is followed by a short dry break in August known as the August break
which is a short dry season lasting for two to three weeks in August . This break is broken by the short rainy season starting around early September and
lasting to Mid October with a peak period at the end of September . The ending of the short rainy season in October is followed by long dry season . This period
starts from late October and lasts till early March with peak dry conditions between early December and late February . The tropical savanna climate or tropical
wet and dry climate , is extensive in area and covers most of Western Nigeria to central Nigeria beginning from the Tropical rainforest climate boundary in
southern Nigeria to the central part of Nigeria , where it exerts enormous influence on the region .
</item 10>
=== END ITEM id=760caed385dc2135 ===

=== ITEM id=3344278138dd41e9 ===
Gold sentences and facts:
Sentence 0: The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016 . Unlike the first six seasons that each had ten episodes and the seventh that had seven episodes , the eighth season will have only six episodes . Like the previous season , it will largely consist of original content not found currently in George R.R. Martin 's A Song of Ice and Fire series , and will instead adapt material Martin has revealed to showrunners about the upcoming novels in the series , The Winds of Winter and A Dream of Spring .
  Fact 0.0: The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016.
  Fact 0.1: The first six seasons of Game of Thrones each had ten episodes.
  Fact 0.2: The seventh season of Game of Thrones had seven episodes.
  Fact 0.3: The eighth season of Game of Thrones will have only six episodes.
  Fact 0.4: Like the previous season, the eighth season will largely consist of original content not found currently in George R.R. Martin's A Song of Ice and Fire series.
  Fact 0.5: The eighth season will instead adapt material Martin has revealed to showrunners about the upcoming novels in the series.
  Fact 0.6: The upcoming novels in the series are The Winds of Winter and A Dream of Spring.

Retrieved items:
<item 1>
Game of Thrones (season 8)
The final episode of the show will be directed by Benioff and Weiss , who have previously directed one episode each . At the show 's South by Southwest panel on
March 12 , 2017 , Benioff and Weiss announced the writers for the show to be Dave Hill ( episode 1 ) and Bryan Cogman ( episode 2 ) . The showrunners will then
divide up the screenplay for the remaining four episodes amongst themselves . Writing ( edit ) Writing for the eighth season started with a 140 - page outline .
Benioff said that the divvying up process and who should write what section became more difficult , being that `` this would be the last time that we would be
doing this . '' Filming ( edit ) In an interview with Entertainment Weekly , HBO programming president Casey Bloys stated that instead of the series finale
being a feature film , the final season would be `` six one - hour movies '' on television . He continued , `` The show has proven that TV is every bit as
impressive and in many cases more so , than film . What they 're doing is monumental . ''
</item 1>

<item 2>
Game of Thrones (season 8)
`` ' Game of Thrones ' Creators Mull Shorter Final Seasons ( EXCLUSIVE ) '' . Variety . Archived from the original on August 21 , 2016 . Retrieved June 27 ,
2016 . Jump up ^ Goldberg , Lesley ( April 14 , 2016 ) . `` ' Game of Thrones ' Considering Shorter Final Seasons '' . The Hollywood Reporter . Archived from
the original on June 18 , 2016 . Retrieved June 27 , 2016 . Jump up ^ Hibberd , James ( July 18 , 2016 ) . `` Game of Thrones : HBO announces summer return , 7
episodes '' . Entertainment Weekly . Archived from the original on September 1 , 2016 . Retrieved July 18 , 2016 . ^ Jump up to : Hibberd , James ( March 12 ,
2017 ) . `` Game of Thrones producers confirm final season only 6 episodes '' . Entertainment Weekly . Archived from the original on March 12 , 2017 . Retrieved
March 13 , 2017 .
</item 2>

<item 3>
Game of Thrones (season 8)
Game of Thrones ( season 8 ) - wikipedia Game of Thrones ( season 8 ) Jump to : navigation , search Game of Thrones ( season 8 ) Promotional poster Starring See
List of Game of Thrones cast Country of origin United States Release Original network HBO Season chronology ← Previous Season 7 List of Game of Thrones episodes
The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016 . Unlike the first six seasons that each
had ten episodes and the seventh that had seven episodes , the eighth season will have only six episodes . Like the previous season , it will largely consist of
original content not found currently in George R.R. Martin 's A Song of Ice and Fire series , and will instead adapt material Martin has revealed to showrunners
about the upcoming novels in the series , The Winds of Winter and A Dream of Spring . The season will be adapted for television by David Benioff and D.B. Weiss
. Filming officially began on October 23 , 2017 . The season is scheduled to premiere in 2019 .
</item 3>

<item 4>
Game of Thrones (season 8)
What they 're doing is monumental . '' Filming officially began on October 23 , 2017 . Episodes ( edit ) Co-creators David Benioff and D.B. Weiss have said that
the seventh and eighth season would likely consist of fewer episodes , stating that after season six , they were `` down to our final 13 episodes after this
season . We 're heading into the final lap . '' Benioff and Weiss stated that they were unable to produce 10 episodes in the show 's usual 12 to 14 month time
frame , as Weiss said , `` It 's crossing out of a television schedule into more of a mid-range movie schedule . '' HBO confirmed in July 2016 , that the
seventh season would consist of seven episodes , and would premiere later than usual in mid-2017 because of the later filming schedule . Benioff and Weiss later
confirmed that the eighth season will consist of six episodes , and is expected to premiere later than usual for the same reason .
</item 4>

<item 5>
Game of Thrones (season 8)
Benioff and Weiss spoke about the end of the show , saying , `` From the beginning we 've wanted to tell a 70 - hour movie . It will turn out to be a 73 - hour
movie , but it 's stayed relatively the same of having the beginning , middle and now we 're coming to the end . It would have been really tough if we lost any
core cast members along the way , I 'm very happy we 've kept everyone and we get to finish it the way we want to . '' The season is set to air in 2019 . Music
( edit ) Ramin Djawadi is set to return as the composer of the show for the eighth season . References ( edit ) Jump up ^ Hibberd , James ( July 30 , 2016 ) .
`` Game of Thrones : HBO confirms season 8 will be last '' . Entertainment Weekly . Archived from the original on September 1 , 2016 . Retrieved March 13 , 2017
. Jump up ^ Birnbaum , Debra ( July 30 , 2016 ) .
</item 5>

<item 6>
Game of Thrones (season 8)
Queen ( 2013 ) The Rogue Prince ( 2014 ) The Sons of the Dragon ( 2017 ) TV series Episodes Season 1 5 6 7 8 Awards Characters Music `` Game of Thrones Theme ''
`` The Rains of Castamere '' `` The Bear and the Maiden Fair '' `` Light of the Seven '' Catch the Throne Live Concert Experience Soundtracks Season 1 5 6 7
Title sequence After the Thrones Thronecast Video games A Game of Thrones : Genesis Game of Thrones ( 2012 ) Game of Thrones Ascent Game of Thrones ( 2014 )
Other media The World of Ice & Fire ( 2014 ) Board game Card game ( Second edition ) Comic book series Characters Petyr Baelish Joffrey Baratheon Renly
Baratheon Robert Baratheon Stannis Baratheon Tommen Baratheon Ramsay Bolton Roose Bolton Bronn Gregor Clegane Sandor Clegane Khal Drogo Gendry Tormund
Giantsbane Gilly Theon Greyjoy Cersei Lannister Jaime Lannister Tyrion Lannister Tywin Lannister Oberyn Martell Melisandre Missandei Jorah
</item 6>

<item 7>
Game of Thrones (season 8)
The season is scheduled to premiere in 2019 . Contents ( hide ) 1 Episodes 2 Cast 2.1 Main cast 2.2 Supporting cast 3 Production 3.1 Crew 3.2 Writing 3.3
Filming 3.4 Episodes 3.5 Music 4 References 5 External links Episodes ( edit ) See also : List of Game of Thrones episodes No . overall No. in season Title
Directed by Written by Original air date U.S. viewers ( millions ) 68 TBA David Nutter Dave Hill 2019 ( 2019 ) TBD 69 TBA David Nutter Bryan Cogman 2019 ( 2019
) TBD 70 TBA Miguel Sapochnik David Benioff & D.B. Weiss 2019 ( 2019 ) TBD 71 TBA David Nutter David Benioff & D.B. Weiss 2019 ( 2019 ) TBD 72 5 TBA Miguel
Sapochnik David Benioff & D.B.
</item 7>

<item 8>
Game of Thrones (season 8)
Jump up ^ Chitwood , Adam ( December 15 , 2017 ) . `` ' Justice League ' DP Fabian Wagner on Zack Snyder 's Cut , Superman 's Black Suit & ' Game of Thrones '
'' . Collider . Archived from the original on December 16 , 2017 . Retrieved December 18 , 2017 . Jump up ^ Carson , Biz ( March 12 , 2017 ) . `` The final
season of ' Game of Thrones ' will only be six episodes '' . Business Insider . Archived from the original on April 12 , 2017 . Retrieved April 11 , 2017 . ^
Jump up to : `` Game of Thrones Will Return in 2019 '' . HBO . Archived from the original on January 4 , 2018 . Retrieved January 4 , 2018 . ^ Jump up to :
Goldberg , Lesley ( June 21 , 2016 ) . `` ' Game of Thrones ' Stars Score Hefty Pay Raises for Season 8 '' . The Hollywood Reporter .
</item 8>

<item 9>
Game of Thrones (season 8)
Weiss 2019 ( 2019 ) TBD 73 6 TBA David Benioff & D.B. Weiss David Benioff & D.B. Weiss 2019 ( 2019 ) TBD Cast ( edit ) See also : List of Game of Thrones
characters Peter Dinklage ( Tyrion Lannister ) Nikolaj Coster - Waldau ( Jaime Lannister ) Lena Headey ( Cersei Lannister ) Emilia Clarke ( Daenerys Targaryen )
Main cast ( edit ) Peter Dinklage as Tyrion Lannister Nikolaj Coster - Waldau as Jaime Lannister Lena Headey as Cersei Lannister Emilia Clarke as Daenerys
Targaryen Kit Harington as Jon Snow Liam Cunningham as Davos Seaworth Sophie Turner as Sansa Stark Maisie Williams as Arya Stark Nathalie Emmanuel as Missandei
Gwendoline Christie as Brienne of Tarth John Bradley as Samwell Tarly Isaac Hempstead Wright as Bran Stark Rory McCann as Sandor `` The Hound '' Clegane Iain
Glen as Jorah Mormont Conleth Hill as Varys
</item 9>

<item 10>
Game of Thrones (season 8)
Jump up ^ Birnbaum , Debra ( July 30 , 2016 ) . `` HBO Confirms ' Game of Thrones ' Will End With Eighth Season '' . Variety . Archived from the original on
September 2 , 2016 . Retrieved March 13 , 2017 . Jump up ^ Hibberd , James ( May 24 , 2016 ) . `` George R.R. Martin revealed 3 huge shocks to Game of Thrones
producers '' . Entertainment Weekly . Archived from the original on August 21 , 2016 . Retrieved March 13 , 2017 . ^ Jump up to : Hooton , Christopher (
September 27 , 2017 ) . `` Game of Thrones season 8 : Directors revealed for all episodes '' . The Independent . Archived from the original on October 20 , 2017
. Retrieved October 20 , 2017 . Jump up ^ Chitwood , Adam ( December 15 , 2017 ) .
</item 10>
=== END ITEM id=3344278138dd41e9 ===

=== ITEM id=20a66e4694d5cd22 ===
Gold sentences and facts:
Sentence 0: A subcutaneous injection is administered as a bolus into the subcutis , the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut is the preferred abbreviation for patient safety .
  Fact 0.0: A subcutaneous injection is administered as a bolus into the subcutis.
  Fact 0.1: The subcutis is the layer of skin directly below the dermis and epidermis.
  Fact 0.2: The dermis and epidermis are collectively referred to as the cutis.
  Fact 0.3: Subcutaneous injections are highly effective in administering vaccines.
  Fact 0.4: Subcutaneous injections are highly effective in administering medications such as insulin, morphine, diacetylmorphine and goserelin.
  Fact 0.5: Subcutaneous injection of recreational drugs, as opposed to intravenous injection, is referred to as 'skin popping'.
  Fact 0.6: Subcutaneous administration may be abbreviated as SC, SQ, sub-cu, sub-Q, SubQ, or subcut.
  Fact 0.7: Subcut is the preferred abbreviation for patient safety.

Retrieved items:
<item 1>
Subcutaneous injection
musculoskeletal Intravenous Intracardiac Intramuscular Intraosseous Intraperitoneal Nanocell injection Patient - Controlled Analgesia pump PIC line WikiProject
Pharmacy portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Subcutaneous_injection&oldid=827949994 '' Categories : Dosage forms Routes of
administration Injection ( medicine ) Talk Contents About Wikipedia Dansk Deutsch Español فارسی Français Italiano Nederlands Norsk Polski Português Slovenščina
Türkçe Українська Tiếng Việt 6 more Edit links This page was last edited on 27 February 2018 , at 18 : 16 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
Subcutaneous injection
otologic , nasal Nasal spray Ear drops Eye drops Ointment Hydrogel Nanosphere suspension Insufflation Mucoadhesive microdisc ( microsphere tablet ) Urogenital
Ointment Pessary ( vaginal suppository ) Vaginal ring Vaginal douche Intrauterine device ( IUD ) Extra-amniotic infusion Intravesical infusion Rectal ( enteral
) Ointment Suppository Enema Solution Hydrogel Murphy drip Nutrient enema Dermal Ointment Topical cream Topical gel Liniment Paste Film DMSO drug solution
Electrophoretic dermal delivery system Hydrogel Liposomes Transfersome vesicles Cream Lotion Lip balm Medicated shampoo Dermal patch Transdermal patch Contact (
rubbed into break in the skin ) Transdermal spray Jet injector Injection , infusion ( into tissue / blood ) Skin Intradermal Subcutaneous Transdermal implant
Organs Intracavernous Intravitreal Intra-articular injection Transscleral Central nervous system Intracerebral Intrathecal Epidural Circulatory ,
</item 2>

<item 3>
Subcutaneous injection
It is slower than intramuscular injections but still faster than intradermal injections . Contents ( hide ) 1 Injection sites 2 Equipment 3 Procedure 4
Subcutaneous insulin injections in diabetes mellitus 5 See also 6 References Injection sites ( edit ) Subcutaneous injection sites Sites include , The outer
area of the upper arm . The abdomen , from the rib margin to the iliac crest and avoiding a 2 - inch circle around the navel . This has the fastest rate of
absorption among the sites . The front of the thigh , midway to the outer side , 4 inches below the top of the thigh to 4 inches above the knee . This has a
slower rate of absorption than the upper arm . The upper back . The upper area of the buttock , just behind the hip bone . This has the slowest rate of
absorption among the sites . Equipment ( edit ) A 25 to 31 gauge thick , 3 / 8 '' to 1 '' long needle can be used . The size is determined by the amount of
subcutaneous tissue present , which is based on patient build .
</item 3>

<item 4>
Subcutaneous injection
Subcutaneous injection - wikipedia Subcutaneous injection Jump to : navigation , search `` Subcut '' redirects here . For the song by Grand Mixer DXT featuring
Bill Laswell , see Aftermathematics . Play media 3D animation of a subcutaneous injection A subcutaneous injection is administered as a bolus into the subcutis
, the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in
administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of
recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut
is the preferred abbreviation for patient safety . Subcutaneous tissue has few blood vessels and so drugs injected here are for slow , sustained rates of
absorption . It is slower than intramuscular injections but still faster than intradermal injections .
</item 4>

<item 5>
Serratus anterior muscle
The muscle is inserted along the medial border of the scapula between the superior and inferior angles along with being inserted along the thoracic vertebrae .
The muscle is divided into three named parts depending on their points of insertions : the serratus anterior superior is inserted near the superior angle the
serratus anterior intermediate is inserted along the medial border the serratus anterior inferior is inserted near the inferior angle . Relations ( edit ) The
serratus anterior lies deep to the subscapularis , from which it is separated by the subscapularis ( supraserratus ) bursa . It is separated from the rib by the
scapulothoracic ( infraserratus ) bursa . Nerve supply ( edit ) The serratus anterior is innervated by the long thoracic nerve ( Nerve of Bell ) , a branch of
the brachial plexus . The long thoracic nerve travels inferiorly on the surface of the serratus . The nerve is especially vulnerable during certain types of
surgery ( for example , during lymph node clearance from the axilla ( e.g.
</item 5>

<item 6>
Subcutaneous injection
The injection should be given under the skin , into the fat layer . See also ( edit ) Intramuscular injection Intravenous injection Intradermal injection
References ( edit ) Jump up ^ `` subcutaneous injection '' at Dorland 's Medical Dictionary Jump up ^ `` ISMP 's List of Error - Prone Abbreviations , Symbols ,
and Dose Designations '' ( PDF ) . www.ismp.org. 2013 . Retrieved 13 May 2013 . ^ Jump up to : Taylor , C.R. , Lillis , C. , LeMone , P. , Lynn , P. ( 2011 )
Fundamentals of nursing : The art and science of nursing care . Philadelphia : Lippincott Williams & Wilkins , page 749 . Jump up ^ `` Insulin Administration ''
. Diabetes Care . 25 ( suppl 1 ) : s112 -- s115 . January 2002 . doi : 10.2337 / diacare. 25.2007. s112 . ( hide ) Routes of administration ,
</item 6>

<item 7>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 7>

<item 8>
Subcutaneous injection
The 3 / 8 '' and 5 / 8 '' needles are most commonly used . Usually , no more than 1 mL of solution is given , compared to intradermal injections , where no more
than 0.5 mL is usually given . Procedure ( edit ) Subcutaneous injections are inserted at 45 to 90 degree angles , depending on amount of subcutaneous tissue
present and length of needle - a shorter , 3 / 8 '' needle is usually inserted 90 degrees and a 5 / 8 '' needle is usually inserted at 45 degrees . Medication
is administered slowly , about 10 seconds / milliliter . Subcutaneous insulin injections in diabetes mellitus ( edit ) An insulin pump with a subcutaneous
injection site . A person with insulin - dependent ( either type 1 or type 2 ) diabetes mellitus typically injects insulin subcutaneously . The injection should
be given under the skin , into the fat layer .
</item 8>

<item 9>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 9>

<item 10>
Subcutaneous injection
25.2007. s112 . ( hide ) Routes of administration , dosage forms Oral Digestive tract ( enteral ) Solids Pill Tablet Capsule Pastille Time release technology
Osmotic delivery system ( OROS ) Liquids Decoction Elixir Electuary Emulsion Extended - release syrup Effervescent powder or tablet Herbal tea Hydrogel
Molecular encapsulation Powder Softgel Solution Suspension Syrup Syrup Concentrate for dilution and / or addition of carbonated water Tincture Buccal (
sublabial ) , sublingual Solids Orally disintegrating tablet ( ODT ) Film Lollipop Sublingual drops Lozenges Effervescent buccal tablet Chewing gum Liquids
Mouthwash Toothpaste Ointment Oral spray Respiratory tract Solids Smoking device Dry - powder inhaler ( DPI ) 0 0 Liquids Anaesthetic vaporizer Vaporizer
Nebulizer Metered - dose inhaler ( MDI ) Gas Oxygen mask and Nasal cannula Oxygen concentrator Anaesthetic machine Relative analgesia machine Ophthalmic ,
otologic ,
</item 10>
=== END ITEM id=20a66e4694d5cd22 ===
