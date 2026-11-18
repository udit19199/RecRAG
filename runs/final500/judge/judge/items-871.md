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

=== ITEM id=21396d48cf71fabd ===
Gold sentences and facts:
Sentence 0: During the 1995 -- 96 season , Alan Shearer became the first player to score 100 Premier League goals , and holds the record for the fewest games taken to reach 100 , doing so in 124 appearances . He also holds the record for most goals scored in the Premier League . After Shearer , Harry Kane is the second - fastest to 100 goals , doing so in 141 games .
  Fact 0.0: During the 1995-96 season, Alan Shearer became the first player to score 100 Premier League goals.
  Fact 0.1: Alan Shearer holds the record for the fewest games taken to reach 100 goals.
  Fact 0.2: Alan Shearer reached 100 goals in 124 appearances.
  Fact 0.3: Alan Shearer holds the record for most goals scored in the Premier League.
  Fact 0.4: Harry Kane is the second-fastest player to reach 100 goals.
  Fact 0.5: Harry Kane reached 100 goals in 141 games.
Sentence 1: List of Premier League players with 100 or more goals Rank Player Position Premier League club ( s ) Goals Played Ratio Shearer , Alan Alan Shearer Blackburn Rovers , Newcastle United 260 441 0.59 Rooney , Wayne Wayne Rooney Everton , Manchester United 208 484 0.43 Cole , Andy Andy Cole Newcastle United , Manchester United , Blackburn Rovers , Fulham , Manchester City , Portsmouth 187 414 0.45 Lampard , Frank Frank Lampard MF West Ham United , Chelsea , Manchester City 177 609 0.29 Henry , Thierry Thierry Henry Arsenal 175 258 0.68 6 Fowler , Robbie Robbie Fowler Liverpool , Leeds United , Manchester City 163 379 0.43 7 Defoe , Jermain Jermain Defoe West Ham United , Portsmouth , Tottenham Hotspur , Sunderland , Bournemouth 162 485 0.33 8 Owen , Michael Michael Owen Liverpool , Newcastle United , Manchester United , Stoke City 150 326 0.46 9 Ferdinand , Les Les Ferdinand Queens Park Rangers , Newcastle United , Tottenham Hotspur , West Ham United , Leicester City , Bolton Wanderers 149 351 0.42 10 Sheringham , Teddy Teddy Sheringham Nottingham Forest , Tottenham Hotspur , Manchester United , Portsmouth , West Ham United 146 418 0.35 11 van Persie , Robin Robin van Persie Arsenal , Manchester United 144 280 0.51 12 Agüero , Sergio Sergio Agüero Manchester City 143 204 0.70 13 Hasselbaink , Jimmy Floyd Jimmy Floyd Hasselbaink Leeds United , Chelsea , Middlesbrough , Charlton Athletic 127 288 0.44 14 Keane , Robbie Robbie Keane Coventry City , Leeds United , Tottenham Hotspur , Liverpool , West Ham United , Aston Villa 126 349 0.36 15 Anelka , Nicolas Nicolas Anelka Arsenal , Liverpool , Manchester City , Bolton Wanderers , Chelsea , West Bromwich Albion 125 364 0.34 16 Yorke , Dwight Dwight Yorke Aston Villa , Manchester United , Blackburn Rovers , Birmingham City , Sunderland 123 375 0.33 17 Gerrard , Steven Steven Gerrard MF Liverpool 120 504 0.24 18 Wright , Ian Ian Wright Arsenal , West Ham United 113 213 0.53 19 Dublin , Dion Dion Dublin Manchester United , Coventry City , Aston Villa 111 312 0.36 20 Heskey , Emile Emile Heskey Leicester City , Liverpool , Birmingham City , Wigan Athletic , Aston Villa 110 516 0.21 21 Giggs , Ryan Ryan Giggs MF Manchester United 109 632 0.17 22 Crouch , Peter Peter Crouch Aston Villa , Southampton , Liverpool , Portsmouth , Tottenham Hotspur , Stoke City 107 454 0.23 22 Scholes , Paul Paul Scholes MF Manchester United 107 499 0.21 24 Bent , Darren Darren Bent Ipswich Town , Charlton Athletic , Tottenham Hotspur , Sunderland , Aston Villa , Fulham 106 276 0.38 25 Kane , Harry Harry Kane Tottenham Hotspur , Norwich City 104 149 0.70 26 Drogba , Didier Didier Drogba Chelsea 104 254 0.41 27 Lukaku , Romelu Romelu Lukaku West Bromwich Albion , Everton , Manchester United 101 216 0.47 28 Le Tissier , Matt Matt Le Tissier MF Southampton 100 270 0.37 Italics show players still playing professional football ; Bold shows players still playing in the Premier League .
  Fact 1.0: The list is of Premier League players with 100 or more goals.
  Fact 1.1: The table columns are rank, player, position, Premier League club(s), goals, games played, and ratio.
  Fact 1.2: Alan Shearer played for Blackburn Rovers and Newcastle United, scoring 260 goals in 441 games (ratio 0.59).
  Fact 1.3: Wayne Rooney played for Everton and Manchester United, scoring 208 goals in 484 games (ratio 0.43).
  Fact 1.4: Andy Cole played for Newcastle United, Manchester United, Blackburn Rovers, Fulham, Manchester City and Portsmouth, scoring 187 goals in 414 games (ratio 0.45).
  Fact 1.5: Frank Lampard (position MF) played for West Ham United, Chelsea and Manchester City, scoring 177 goals in 609 games (ratio 0.29).
  Fact 1.6: Thierry Henry played for Arsenal, scoring 175 goals in 258 games (ratio 0.68).
  Fact 1.7: Rank 6: Robbie Fowler played for Liverpool, Leeds United and Manchester City, scoring 163 goals in 379 games (ratio 0.43).
  Fact 1.8: Rank 7: Jermain Defoe played for West Ham United, Portsmouth, Tottenham Hotspur, Sunderland and Bournemouth, scoring 162 goals in 485 games (ratio 0.33).
  Fact 1.9: Rank 8: Michael Owen played for Liverpool, Newcastle United, Manchester United and Stoke City, scoring 150 goals in 326 games (ratio 0.46).
  Fact 1.10: Rank 9: Les Ferdinand played for Queens Park Rangers, Newcastle United, Tottenham Hotspur, West Ham United, Leicester City and Bolton Wanderers, scoring 149 goals in 351 games (ratio 0.42).
  Fact 1.11: Rank 10: Teddy Sheringham played for Nottingham Forest, Tottenham Hotspur, Manchester United, Portsmouth and West Ham United, scoring 146 goals in 418 games (ratio 0.35).
  Fact 1.12: Rank 11: Robin van Persie played for Arsenal and Manchester United, scoring 144 goals in 280 games (ratio 0.51).
  Fact 1.13: Rank 12: Sergio Aguero played for Manchester City, scoring 143 goals in 204 games (ratio 0.70).
  Fact 1.14: Rank 13: Jimmy Floyd Hasselbaink played for Leeds United, Chelsea, Middlesbrough and Charlton Athletic, scoring 127 goals in 288 games (ratio 0.44).
  Fact 1.15: Rank 14: Robbie Keane played for Coventry City, Leeds United, Tottenham Hotspur, Liverpool, West Ham United and Aston Villa, scoring 126 goals in 349 games (ratio 0.36).
  Fact 1.16: Rank 15: Nicolas Anelka played for Arsenal, Liverpool, Manchester City, Bolton Wanderers, Chelsea and West Bromwich Albion, scoring 125 goals in 364 games (ratio 0.34).
  Fact 1.17: Rank 16: Dwight Yorke played for Aston Villa, Manchester United, Blackburn Rovers, Birmingham City and Sunderland, scoring 123 goals in 375 games (ratio 0.33).
  Fact 1.18: Rank 17: Steven Gerrard (position MF) played for Liverpool, scoring 120 goals in 504 games (ratio 0.24).
  Fact 1.19: Rank 18: Ian Wright played for Arsenal and West Ham United, scoring 113 goals in 213 games (ratio 0.53).
  Fact 1.20: Rank 19: Dion Dublin played for Manchester United, Coventry City and Aston Villa, scoring 111 goals in 312 games (ratio 0.36).
  Fact 1.21: Rank 20: Emile Heskey played for Leicester City, Liverpool, Birmingham City, Wigan Athletic and Aston Villa, scoring 110 goals in 516 games (ratio 0.21).
  Fact 1.22: Rank 21: Ryan Giggs (position MF) played for Manchester United, scoring 109 goals in 632 games (ratio 0.17).
  Fact 1.23: Rank 22: Peter Crouch played for Aston Villa, Southampton, Liverpool, Portsmouth, Tottenham Hotspur and Stoke City, scoring 107 goals in 454 games (ratio 0.23).
  Fact 1.24: Rank 22: Paul Scholes (position MF) played for Manchester United, scoring 107 goals in 499 games (ratio 0.21).
  Fact 1.25: Rank 24: Darren Bent played for Ipswich Town, Charlton Athletic, Tottenham Hotspur, Sunderland, Aston Villa and Fulham, scoring 106 goals in 276 games (ratio 0.38).
  Fact 1.26: Rank 25: Harry Kane played for Tottenham Hotspur and Norwich City, scoring 104 goals in 149 games (ratio 0.70).
  Fact 1.27: Rank 26: Didier Drogba played for Chelsea, scoring 104 goals in 254 games (ratio 0.41).
  Fact 1.28: Rank 27: Romelu Lukaku played for West Bromwich Albion, Everton and Manchester United, scoring 101 goals in 216 games (ratio 0.47).
  Fact 1.29: Rank 28: Matt Le Tissier (position MF) played for Southampton, scoring 100 goals in 270 games (ratio 0.37).
  Fact 1.30: Italics mark players still playing professional football.
  Fact 1.31: Bold marks players still playing in the Premier League.

Retrieved items:
<item 1>
List of Premier League players with 100 or more goals
During the 1995 -- 96 season , Alan Shearer became the first player to score 100 Premier League goals , and holds the record for the fewest games taken to reach
100 , doing so in 124 appearances . He also holds the record for most goals scored in the Premier League . After Shearer , Harry Kane is the second - fastest to
100 goals , doing so in 141 games . Players ( edit ) As of 18 April 2018 . Sources : List of Premier League players with 100 or more goals Rank Player Position
Premier League club ( s ) Goals Played Ratio Shearer , Alan Alan Shearer Blackburn Rovers , Newcastle United 260 441 0.59 Rooney , Wayne Wayne Rooney Everton ,
Manchester United 208 484 0.43 Cole , Andy Andy Cole Newcastle United , Manchester United , Blackburn Rovers , Fulham , Manchester City , Portsmouth 187 414
0.45 Lampard , Frank Frank Lampard MF West Ham United , Chelsea , Manchester City 177 609 0.29 Henry , Thierry Thierry Henry Arsenal 175 258 0.68 6 Fowler ,
Robbie Robbie Fowler Liverpool ,
</item 1>

<item 2>
Graph query row:
player: Wayne Rooney
relationship: RELATED_TO
related_entity: Alan Shearer
fact: Wayne Rooney was behind only Alan Shearer in Premier League goals of 200 or more
context: None
result: None
outcome: None
date: None
score: None
</item 2>

<item 3>
List of Premier League players with 100 or more goals
gulf Associated competitions Asia Trophy FA Community Shield FA Cup EFL Cup UEFA Champions League UEFA Europa League Premier League 2 U18 Premier League Premier
League International Cup Retrieved from `` https://en.wikipedia.org/w/index.php?title=List_of_Premier_League_players_with_100_or_more_goals&oldid=837185393 ''
Categories : Lists of Premier League players Association football records and statistics Hidden categories : Wikipedia pending changes protected pages Use dmy
dates from March 2012 Articles with hCards Talk About Wikipedia Bosanski Dansk Español Italiano Shqip Українська Edit links This page was last edited on 19
April 2018 , at 07 : 24 . About Wikipedia
</item 3>

<item 4>
List of Premier League players with 100 or more goals
68 6 Fowler , Robbie Robbie Fowler Liverpool , Leeds United , Manchester City 163 379 0.43 7 Defoe , Jermain Jermain Defoe West Ham United , Portsmouth ,
Tottenham Hotspur , Sunderland , Bournemouth 162 485 0.33 8 Owen , Michael Michael Owen Liverpool , Newcastle United , Manchester United , Stoke City 150 326
0.46 9 Ferdinand , Les Les Ferdinand Queens Park Rangers , Newcastle United , Tottenham Hotspur , West Ham United , Leicester City , Bolton Wanderers 149 351
0.42 10 Sheringham , Teddy Teddy Sheringham Nottingham Forest , Tottenham Hotspur , Manchester United , Portsmouth , West Ham United 146 418 0.35 11 van Persie
, Robin Robin van Persie Arsenal , Manchester United 144 280 0.51 12 Agüero , Sergio Sergio Agüero Manchester City 143 204 0.70 13 Hasselbaink , Jimmy Floyd
Jimmy Floyd Hasselbaink Leeds United , Chelsea , Middlesbrough , Charlton Athletic 127 288 0.44 14 Keane ,
</item 4>

<item 5>
List of Premier League players with 100 or more goals
Charlton Athletic 127 288 0.44 14 Keane , Robbie Robbie Keane Coventry City , Leeds United , Tottenham Hotspur , Liverpool , West Ham United , Aston Villa 126
349 0.36 15 Anelka , Nicolas Nicolas Anelka Arsenal , Liverpool , Manchester City , Bolton Wanderers , Chelsea , West Bromwich Albion 125 364 0.34 16 Yorke ,
Dwight Dwight Yorke Aston Villa , Manchester United , Blackburn Rovers , Birmingham City , Sunderland 123 375 0.33 17 Gerrard , Steven Steven Gerrard MF
Liverpool 120 504 0.24 18 Wright , Ian Ian Wright Arsenal , West Ham United 113 213 0.53 19 Dublin , Dion Dion Dublin Manchester United , Coventry City , Aston
Villa 111 312 0.36 20 Heskey , Emile Emile Heskey Leicester City , Liverpool , Birmingham City , Wigan Athletic , Aston Villa 110 516 0.21 21 Giggs , Ryan Ryan
Giggs MF Manchester United 109 632 0.17 22 Crouch ,
</item 5>

<item 6>
List of Premier League players with 100 or more goals
PremierLeague.com . Premier League . Retrieved 10 December 2017 . Premier League Seasons 1992 -- 93 1993 -- 94 1994 -- 95 1995 -- 96 1996 -- 97 1997 -- 98 1998
-- 99 1999 -- 2000 2000 -- 01 2001 -- 02 2002 -- 03 2003 -- 04 2004 -- 05 2005 -- 06 2006 -- 07 2007 -- 08 2008 -- 09 2009 -- 10 2010 -- 11 2011 -- 12 2012 --
13 2013 -- 14 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 Clubs 2017 -- 18 AFC Bournemouth Arsenal Brighton & Hove Albion Burnley Chelsea Crystal
Palace Everton Huddersfield Town Leicester City Liverpool Manchester City Manchester United Newcastle United Southampton Stoke City Swansea City Tottenham
Hotspur Watford West Bromwich Albion West Ham
</item 6>

<item 7>
List of Premier League players with 100 or more goals
Ryan Ryan Giggs MF Manchester United 109 632 0.17 22 Crouch , Peter Peter Crouch Aston Villa , Southampton , Liverpool , Portsmouth , Tottenham Hotspur , Stoke
City 107 454 0.23 22 Scholes , Paul Paul Scholes MF Manchester United 107 499 0.21 24 Bent , Darren Darren Bent Ipswich Town , Charlton Athletic , Tottenham
Hotspur , Sunderland , Aston Villa , Fulham 106 276 0.38 25 Kane , Harry Harry Kane Tottenham Hotspur , Norwich City 104 149 0.70 26 Drogba , Didier Didier
Drogba Chelsea 104 254 0.41 27 Lukaku , Romelu Romelu Lukaku West Bromwich Albion , Everton , Manchester United 101 216 0.47 28 Le Tissier , Matt Matt Le
Tissier MF Southampton 100 270 0.37 Italics show players still playing professional football ; Bold shows players still playing in the Premier League .
</item 7>

<item 8>
List of Premier League players with 100 or more goals
Bold shows players still playing in the Premier League . See also ( edit ) English football first tier top scorers List of top Premier League goal scorers by
season List of Premier League players with 500 or more appearances References ( edit ) Jump up ^ `` Bent poised to join 100 club '' . PremierLeague.com .
Premier League . 30 January 2012 . Archived from the original on 20 October 2013 . Retrieved 2 February 2012 . Jump up ^ `` Bent delighted to join 100 goals
club '' . PremierLeague.com . Premier League . 2 February 2012 . Retrieved 2 February 2012 . Jump up ^ Dan Kilpatrick ( 4 February 2018 ) . `` Tottenham striker
Harry Kane scores 100th Premier League goal '' . ESPN . ^ Jump up to : `` Premier League Player Stats : Goals '' . PremierLeague.com . Premier League .
Retrieved 10 December 2017 . Jump up ^ `` Premier League Players -- Overview and Stats '' . PremierLeague.com . Premier League .
</item 8>

<item 9>
List of Premier League players with 100 or more goals
List of Premier League players with 100 or more goals - wikipedia List of Premier League players with 100 or more goals This is the latest accepted revision ,
reviewed on 19 April 2018 . Jump to : navigation , search See also : List of footballers in England and Scotland by number of league goals Alan Shearer was the
first player to score 200 Premier League goals , and only player to score 100 goals for two clubs . Wayne Rooney is the second Premier League player to score
200 goals or more , behind only Shearer and the only active player as of 2018 to do so . Frank Lampard is the Premier League 's highest scoring midfielder of
all time . Thierry Henry scored 175 Premier League goals , the most by a foreign player . Ryan Giggs is the only player to have scored in each of the first 21
Premier League seasons . Didier Drogba is the only African player to score 100 Premier League goals . Since the Premier League 's formation at the start of the
1992 -- 93 season , 28 players have managed to accrue 100 or more goals in the competition .
</item 9>

<item 10>
List of Premier League players with 100 or more goals
Newcastle United Southampton Stoke City Swansea City Tottenham Hotspur Watford West Bromwich Albion West Ham United Former Aston Villa Barnsley Birmingham City
Blackburn Rovers Blackpool Bolton Wanderers Bradford City Cardiff City Charlton Athletic Coventry City Derby County Fulham Hull City Ipswich Town Leeds United
Middlesbrough Norwich City Nottingham Forest Oldham Athletic Portsmouth Queens Park Rangers Reading Sheffield United Sheffield Wednesday Sunderland Swindon Town
Wigan Athletic Wimbledon ( defunct ) Wolverhampton Wanderers ( returning in 2018 - 19 ) Competition Teams winners Players foreign foreign scorers winners
Managers Stadia Founding Broadcasting foundation Broadcasters Referees Statistics and awards Records All - time table Hat - tricks Highest scores Golden Boot
Golden Glove Manager of the Season Player of the Season Manager of the Month Player of the Month Goal of the Month Players with 500 + appearances Players with
100 + goals Top scorers by season Goalkeepers with 100 + clean sheets 10 Seasons Awards 20 Seasons Awards Finances Richest clubs : Deloitte list Forbes ' list
Team owners Transfer records Game 39 Premier League -- Football League gulf Associated competitions Asia Trophy FA Community Shield FA Cup EFL Cup UEFA
Champions League UEFA Europa League
</item 10>
=== END ITEM id=21396d48cf71fabd ===

=== ITEM id=4bc03c1b366145df ===
Gold sentences and facts:
Sentence 0: It was suggested that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end to this project , and the walls of the dome were whitewashed . Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 ) of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco . During the restoration work , which ended in 1995 , the entire pictorial cycle of The Last Judgment was photographed with specially designed equipment and all the information collected in a catalogue . All the restoration information along with reconstructed images of the frescos were stored and managed in the Thesaurus Florentinus computer system .
  Fact 0.0: It was suggested that the interior of the 45 metre (147 ft) wide dome should be covered with a mosaic decoration.
  Fact 0.1: The mosaic decoration was suggested to make the most of the available light coming through the circular windows of the drum.
  Fact 0.2: The mosaic decoration was suggested to make the most of the available light coming through the lantern.
  Fact 0.3: Brunelleschi had proposed that the vault glimmer with resplendent gold.
  Fact 0.4: Brunelleschi died in 1446.
  Fact 0.5: Brunelleschi's death in 1446 put an end to the gold vault project.
  Fact 0.6: The walls of the dome were whitewashed.
  Fact 0.7: Grand Duke Cosimo I de' Medici decided to have the dome painted with a representation of The Last Judgment.
  Fact 0.8: The painting of The Last Judgment covers 3,600 square metres (38,750 square feet) of painted surface.
  Fact 0.9: The painting of The Last Judgment was started in 1568 by Giorgio Vasari and Federico Zuccari.
  Fact 0.10: The painting of The Last Judgment would last until 1579.
  Fact 0.11: The upper portion of the painting, near the lantern, represents The 24 Elders of Apoc. 4.
  Fact 0.12: Vasari finished the upper portion of the painting before his death in 1574.
  Fact 0.13: Federico Zuccari and a number of collaborators, such as Domenico Cresti, finished the other portions of the painting.
  Fact 0.14: The other portions, from top to bottom, are Choirs of Angels; Christ, Mary and Saints; Virtues, Gifts of the Holy Spirit and Beatitudes; and at the bottom of the cupola, Capital Sins and Hell.
  Fact 0.15: The frescoes are considered Zuccari's greatest work.
  Fact 0.16: The quality of the work is uneven because of the input of different artists and the different techniques.
  Fact 0.17: Vasari had used true fresco.
  Fact 0.18: Zuccari had painted in secco.
  Fact 0.19: The restoration work ended in 1995.
  Fact 0.20: During the restoration work, the entire pictorial cycle of The Last Judgment was photographed with specially designed equipment.
  Fact 0.21: All the information collected during the restoration was put into a catalogue.
  Fact 0.22: All the restoration information, along with reconstructed images of the frescos, was stored and managed in the Thesaurus Florentinus computer system.

Retrieved items:
<item 1>
Vasari painted Florence Cathedral’s Last Judgment in true fresco, while Zuccari used secco. Restoration, completed in 1995, documented the cycle with specially
designed photography; its records and reconstructed images were managed in the Thesaurus Florentinus system. Excavations from 1965–1974, reconstructed by
Franklin Toker, revealed Roman houses, an early Christian pavement, and remains of Santa Reparata and its expansions. The public crypt contains Brunelleschi’s
</item 1>

<item 2>
Florence Cathedral
Above this shrine is the painting Last Supper by the lesser - known Giovanni Balducci . There was also a glass - paste mosaic panel The Bust of Saint Zanobius
by the 16th - century miniaturist Monte di Giovanni , but it is now on display in the Museum Opera del Duomo . Many decorations date from the 16th - century
patronage of the Grand Dukes , such as the pavement in colored marble , attributed to Baccio d'Agnolo and Francesco da Sangallo ( 1520 -- 26 ) . Some pieces of
marble from the façade were used , topside down , in the flooring ( as was shown by the restoration of the floor after the 1966 flooding ) . It was suggested
that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the
circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end
to this project , and the walls of the dome were whitewashed .
</item 2>

<item 3>
Florence Cathedral contains Tino da Camaino’s 1323 tomb of Bishop Antonio d’Orso, Benedetto da Maiano’s monumental crucifix, and Bandinelli’s choir enclosure.
Luca della Robbia made the sacristy’s bronze doors and two glazed terracottas. Ghiberti’s silver shrine of Saint Zanobius includes a panel depicting the revival
of a dead child; Giovanni Balducci’s *Last Supper* hangs above it. Monte di
</item 3>

<item 4>
Florence Cathedral
As this cathedral was built with funds from the public , some important works of art in this church honour illustrious men and military leaders of Florence :
Lorenzo Ghiberti had a large artistic impact on the cathedral . Ghiberti worked with Filippo Brunelleschi on the cathedral for eighteen years and had a large
number of projects on almost the whole east end . Some of his works were the stained glass designs , the bronze shrine of Saint Zenobius and marble revetments
on the outside of the cathedral . Dante Before the City of Florence by Domenico di Michelino ( 1465 ) . This painting is especially interesting because it shows
us , apart from scenes of the Divine Comedy , a view on Florence in 1465 , a Florence such as Dante himself could not have seen in his time . Funerary Monument
to Sir John Hawkwood by Paolo Uccello ( 1436 ) . This almost monochrome fresco , transferred to canvas in the 19th century , is painted in terra verde , a color
closest to the patina of bronze . Equestrian statue of Niccolò da Tolentino by Andrea del Castagno ( 1456 ) .
</item 4>

<item 5>
Florence Cathedral contains a 19th-century canvas transfer of a richly decorated, marble-like fresco depicting a condottiere triumphantly on horseback. Like a
companion fresco, it conveys movement, but both painters mishandled foreshortened perspective by using separate vanishing points for horse and pedestal. The
cathedral also has 15th- and 16th-century busts of Giotto, Brunelleschi, Marsilio Ficino, and organist Antonio Squ
</item 5>

<item 6>
Florence Cathedral
Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 )
of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing
The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished
the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom
of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of
different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco .
</item 6>

<item 7>
Florence Cathedral
This fresco , transferred on canvas in the 19th century , in the same style as the previous one , is painted in a color resembling marble . However , it is more
richly decorated and gives more the impression of movement . Both frescoes portray the condottieri as heroic figures riding triumphantly . Both painters had
problems when applying in painting the new rules of perspective to foreshortening : they used two unifying points , one for the horse and one for the pedestal ,
instead a single unifying point . Busts of Giotto ( by Benedetto da Maiano ) , Brunelleschi ( by Buggiano - 1447 ) , Marsilio Ficino , and Antonio Squarcialupi
( a most famous organist ) . These busts all date from the 15th and the 16th centuries . Above the main door is the colossal clock face with fresco portraits of
four Prophets or Evangelists by Paolo Uccello ( 1443 ) . This one - handed liturgical clock shows the 24 hours of the hora italica ( Italian time ) , a period
of time ending with sunset at 24 hours .
</item 7>

<item 8>
Florence Cathedral
Unsourced material may be challenged and removed . ( July 2012 ) ( Learn how and when to remove this template message ) Interior of the cathedral Huge clock
decorated by Paolo Uccello . Dante and the Divine Comedy . Trompe l'oeil of Niccolò da Tolentino . Vasari 's fresco begun in 1568 , and completed by Federico
Zuccari in 1579 . Detail under the dome . The Last Judgement ( detail ) under the dome . Tomb of Antonio d'Orso by Tino da Camaino . The Gothic interior is vast
and gives an empty impression . The relative bareness of the church corresponds with the austerity of religious life , as preached by Girolamo Savonarola . Many
decorations in the church have been lost in the course of time , or have been transferred to the Museum Opera del Duomo , such as the magnificent cantorial
pulpits ( the singing galleries for the choristers ) of Luca della Robbia and Donatello .
</item 8>

<item 9>
Florence Cathedral’s vast Gothic interior is notably austere, reflecting Savonarola’s teachings; many artworks, including singing galleries by Luca della Robbia
and Donatello, were lost or moved to the Opera del Duomo Museum. At Cosimo I de’ Medici’s commission, Vasari and Federico Zuccari painted the 3,600-square-metre
dome with *The Last Judgment* (1568–1579). Vasari completed its upper section before his death in
</item 9>

<item 10>
Florence Cathedral, funded publicly, honors notable Florentines and military leaders. Lorenzo Ghiberti worked with Filippo Brunelleschi for 18 years,
contributing stained-glass designs, Saint Zenobius’s bronze shrine, and exterior marble revetments. Domenico di Michelino’s 1465 painting depicts Dante and the
Divine Comedy alongside a view of Florence as it appeared in 1465. Other works include Paolo Uccello’s 1436 terra verde funerary monument
</item 10>
=== END ITEM id=4bc03c1b366145df ===

=== ITEM id=75e0f9b75c0ecd9a ===
Gold sentences and facts:
Sentence 0: Each state controls where it administers its bar exam . Because the MBE ( below ) is a standardized test , it must be administered on the same day across the country . That day occurs twice a year as the last Wednesday in July and the last Wednesday in February . Two states , Delaware and North Dakota , may administer their bar exams only once , in July , if they do not have enough applicants to merit a second sitting . North Dakota requires ten applicants in order to administer the February exam . Most bar exams are administered on consecutive days . Louisiana is the exception , with the Louisiana Bar Exam being a three - day examination on Monday , Wednesday and Friday . Also , Louisiana 's examination is the longest in the country in terms of examination time , with seven hours on Monday and Wednesday and seven and one half hours on Friday for a total of 21.5 hours of testing . Montana 's bar examination also occurs over a three - day period , with a total of 18 hours of testing . The bar exams in Delaware , Nevada , Ohio , South Carolina , and Texas are also three days long . The California bar exam moved to a two - day format beginning with the July , 2017 exam .
  Fact 0.0: Each state controls where it administers its bar exam.
  Fact 0.1: The MBE is a standardized test.
  Fact 0.2: The MBE must be administered on the same day across the country.
  Fact 0.3: The MBE day occurs twice a year.
  Fact 0.4: The MBE day is the last Wednesday in July and the last Wednesday in February.
  Fact 0.5: Delaware and North Dakota may administer their bar exams only once, in July.
  Fact 0.6: Delaware and North Dakota may do this if they do not have enough applicants to merit a second sitting.
  Fact 0.7: North Dakota requires ten applicants in order to administer the February exam.
  Fact 0.8: Most bar exams are administered on consecutive days.
  Fact 0.9: Louisiana is the exception to consecutive-day bar exams.
  Fact 0.10: The Louisiana Bar Exam is a three-day examination on Monday, Wednesday and Friday.
  Fact 0.11: Louisiana's examination is the longest in the country in terms of examination time.
  Fact 0.12: Louisiana's examination has seven hours on Monday and Wednesday.
  Fact 0.13: Louisiana's examination has seven and one half hours on Friday.
  Fact 0.14: Louisiana's examination totals 21.5 hours of testing.
  Fact 0.15: Montana's bar examination occurs over a three-day period.
  Fact 0.16: Montana's bar examination totals 18 hours of testing.
  Fact 0.17: The bar exams in Delaware, Nevada, Ohio, South Carolina, and Texas are three days long.
  Fact 0.18: The California bar exam moved to a two-day format beginning with the July 2017 exam.
Sentence 1: To refresh their memory on `` black - letter rules '' tested on the bar , most students engage in a regimen of study ( called `` bar review '' ) between graduating from law school and sitting for the bar . For bar review , most students in the United States attend a private bar review course which is provided by a third - party company and not their law school .
  Fact 1.0: Most students engage in a regimen of study called bar review.
  Fact 1.1: Bar review is done to refresh memory on black-letter rules tested on the bar.
  Fact 1.2: Bar review is done between graduating from law school and sitting for the bar.
  Fact 1.3: For bar review, most students in the United States attend a private bar review course.
  Fact 1.4: The private bar review course is provided by a third-party company.
  Fact 1.5: Most students in the United States do not take their bar review course from their law school.

Retrieved items:
<item 1>
Bar examination
Controversy ( edit ) Arguments against bar exams ( edit ) A statement by the Society of American Law Teachers ( SALT ) articulates many criticisms of the bar
exam. . The SALT statement , however , does propose some alternative methods of bar admission that are partially test - based . A response to the SALT statement
was made by Suzanne Darrow - Kleinhau in The Bar Examiner Arguments for alternatives to the bar exam ( edit ) The NCBE published an article in 2005 addressing
alternatives to the bar exam , including a discussion of the Daniel Webster Scholar Honors Program , an alternate certification program introduced at the
University of New Hampshire School of Law ( formerly Franklin Pierce Law Center ) in that year . See also ( edit ) Bar Review Bar association IRAC for essay
tips and strategies References ( edit ) Jump up ^ `` OAB registra recorde de aprovação no 5o Exame de Ordem : quase 25 % - Educação - iG '' .
Ultimosegundo.ig.com.br . Archived from the original on 2016 - 03 - 04 . Retrieved 2017 - 01 - 23 .
</item 1>

<item 2>
Bar examination
This exam is composed of three parts : Criminal Law , Criminal Procedural Law and Law of Criminal Enforcement Civil Law , Civil Procedural Law and Business Law
Constitutional Law , Administrative Law , Labor Law , Social Security Law and Law of the European Union . After passing these exams the candidate can practice
law as an attorney - at - law / barrister or as a secretary / judge at the court or as a prosecutor at the public prosecutor 's office or in - house legal
counsel or may operate individually at any field of law . Ireland ( edit ) The bar exams in Ireland are the preserve of the Honorable Society of King 's Inns ,
which runs a series of fourteen exams over ten weeks , from March to June each year , for those enrolled as students in its one - year Barrister - at - Law
degree course . These exams cover such skills as advocacy , research and opinion writing , consulting with clients , negotiation , drafting of legal documents
and knowledge of civil and criminal procedure . For those who fail to meet the requisite 50 % pass mark , repeats are held in the following August and September
.
</item 2>

<item 3>
Bar examination
The new law school system that began in 2009 allows only the graduates of a law school to apply for the bar exam . Malaysia ( edit ) Main article : Certificate
in Legal Practice ( Malaysia ) Philippines ( edit ) The Philippine Bar Examination is administered once every year on the four Sundays of November ( September
before 2011 ) . It covers eight areas of law : political law , labor law and social legislation , criminal law , civil law , commercial law , taxation law ,
remedial law , and legal ethics and practical exercises . Poland ( edit ) In Poland , the bar examination is taken after graduating from a law faculty at a
university . It allows a person to undertake practice , the duration of which varies depending on the specialization . After the practical period applicants
must pass the exam held by the Professional Chambers with assistance from some members of the Ministry of Justice . Singapore ( edit ) The Singapore Bar
Examination ( Part B) is administered once every year , usually over the course of four days . The exam is generally held on the last week of November , and is
administered by the Singapore Institute of Legal Education ( SILE ) .
</item 3>

<item 4>
Bar examination
The MEE and MPT , as uniform though not standardized tests , also must be administered on the same day across the country -- specifically the day before the MBE
. Preparation for the exam ( edit ) Most law schools teach students common law and how to analyze hypothetical fact patterns like a lawyer , but do not
specifically prepare law students for any particular bar exam . Only a minority of law schools offer bar preparation courses . To refresh their memory on ``
black - letter rules '' tested on the bar , most students engage in a regimen of study ( called `` bar review '' ) between graduating from law school and
sitting for the bar . For bar review , most students in the United States attend a private bar review course which is provided by a third - party company and
not their law school . Multistate standardized Examinations ( edit ) The National Conference of Bar Examiners ( NCBE ) is a U.S. based non-profit organization
that develops national ( `` multistate '' ) standardized tests for admission to the bar in individual states . The organization was founded in 1931 .
</item 4>

<item 5>
Bar examination
A logbook signed by the judge on the bench has to certify their weekly attendance . By the end of the eighteenth month , they are eligible to apply to take the
Final Bar Exam by submitting their case summaries , the logbook and a research work pre-approved by the Bar . It is noteworthy , however , that during these
eighteen months , Trainees are eligible to have a limited practice of law under the supervision of their supervising Attorney . This practice does not include
Supreme Court eligible cases and certain criminal and civil cases . Candidates will be tested on Civil law , Civil Procedure , Criminal law , Criminal Procedure
, Commercial Law , Notary ( including rules pertaining Official Documents , Land & Real Estate registrations and regulations etc . ) . Each exam takes two days
, a day on oral examination in front of a judge or an attorney , and a day of essay examination , in which they will be tested on hypothetical cases submitted
to them . Successful applicants will be honoured with the title of `` First Degree Attorney '' , after they take the oath and can practice in all courts of the
country including the Supreme Court . Those who fail must redo the apprenticeship program in full or in part before taking the Final Bar Exam again .
</item 5>

<item 6>
Bar examination
The common law provinces all require prospective lawyers to complete a term of articles ( usually 10 months ) after graduation from law school during which they
work under the supervision of a qualified lawyer . The bar exams may be taken after graduation from law school , but before the commencement of articling , or
may be taken during or after articling is completed . Once the barristers ' and solicitors ' exams have been passed and the term of articles is successfully
completed , students may then be called the bar and admitted to the legal profession as lawyers ( barristers and solicitors ) . People 's Republic of China (
edit ) Main article : State Judicial Exam England and Wales ( edit ) Since the UK has a separated legal profession , Law graduates in England and Wales can take
examinations to qualify as a Barrister or a Solicitor by either undertaking the BPTC or the LPC respectively . These courses are the vocational part of the
training required under the Bar Association and The Law Society rules and are either undertaken on a full - time basis for one year or on a part - time basis
over two years .
</item 6>

<item 7>
Bar examination
Retrieved 2 May 2015 . Jump up ^ 54 JOURNAL OF LEGAL EDUCATION 442 -- 458 Jump up ^ Suzanne Darrow - Kleinhau , `` A Response to Criticism of the Bar Exam '' ,
The Bar Examiner , May 2005 `` Archived copy '' ( PDF ) . Archived from the original on December 3 , 2010 . Retrieved 2011 - 02 - 17 . CS1 maint : BOT :
original - url status unknown ( link ) External links ( edit ) National Conference of Bar Examiners : Makers of the Multistate Bar Exam The Louisiana Supreme
Court Committee on Bar Admission NDL : 00570925 Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Bar examination
The exam is divided in two stages -- the first consists of 80 multiple choice questions covering all disciplines ( Ethics , Human Rights , Philosophy of Law ,
Constitutional Law , Administrative Law , Civil Law , Consumer Law , Civil Procedure Law , Criminal Law , Criminal Procedure Law , Labour Law and Labour
Procedure Law ) . The candidate must score at least 40 questions correctly to proceed to the second part of the exam , four essay questions and a drafting
project ( motion , opinion or claim document ) in Civil Law ( including Consumer Law ) , Labour Law , Criminal Law , Administrative Law , Constitutional Law ,
Corporate Law or Tax Law , and their respective procedures . The Bar examination can be taken on the graduation year . Success in the examination allows one to
practice in any court or jurisdiction of the country . Canada ( edit ) In Canada , admission to the bar is a matter of provincial or territorial jurisdiction .
All provinces except for Québec follow a common law tradition . Lawyers in every common law province are qualified as both barristers and solicitors , and must
pass a Barristers ' Examination and Solicitors ' Examination administered by the Law Society that governs the legal profession in their respective province or
territory .
</item 8>

<item 9>
Bar examination
Because the MBE ( below ) is a standardized test , it must be administered on the same day across the country . That day occurs twice a year as the last
Wednesday in July and the last Wednesday in February . Two states , Delaware and North Dakota , may administer their bar exams only once , in July , if they do
not have enough applicants to merit a second sitting . North Dakota requires ten applicants in order to administer the February exam . Most bar exams are
administered on consecutive days . Louisiana is the exception , with the Louisiana Bar Exam being a three - day examination on Monday , Wednesday and Friday .
Also , Louisiana 's examination is the longest in the country in terms of examination time , with seven hours on Monday and Wednesday and seven and one half
hours on Friday for a total of 21.5 hours of testing . Montana 's bar examination also occurs over a three - day period , with a total of 18 hours of testing .
The bar exams in Delaware , Nevada , Ohio , South Carolina , and Texas are also three days long . The California bar exam moved to a two - day format beginning
with the July , 2017 exam .
</item 9>

<item 10>
Bar examination
Criminal law , employment law , constitution law , administrative law , and tax law . Civil procedure law , bankruptcy and business reorganization law , and the
system of the court of justice . And Criminal procedure law , human rights , and law on the evidence . Each part has 10 essay questions . The pass mark is 50 .
The parts 1 - 2 are usually taken in October and the rest are usually taken in March . One does not need to pass all four parts in one year . After passing all
the written exams , there is an oral exam . Around 10,000 bar students sit the exam each year . In 2013 , 1,231 students were called to the bar , 111 of whom
did it in only one year . Quite confusingly with international norms , students called to the bar are referred to as netibandit ( เนติบัณฑิต ) , which gets
translated into English as Barrister - at - Law . The Thai legal profession , however , is a fused one and those with lawyer licenses are able to practice both
as barristers and solicitors in the British / Commonwealth sense .
</item 10>
=== END ITEM id=75e0f9b75c0ecd9a ===

=== ITEM id=6db45cd98c2a2ca0 ===
Gold sentences and facts:
Sentence 0: Reading Full name Reading Football Club Nickname ( s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season
  Fact 0.0: The full name of the club is Reading Football Club.
  Fact 0.1: The club's nickname is The Royals.
  Fact 0.2: Reading was founded in 1871.
  Fact 0.3: Reading was founded 147 years ago.
  Fact 0.4: Reading's ground is Madejski Stadium, with a ground capacity of 24,161.
  Fact 0.5: Reading's owners are Dai Yongge and Dai Xiuli (majority).
  Fact 0.6: Reading's chairman is Sir John Madejski.
  Fact 0.7: Reading's manager is Jaap Stam.
  Fact 0.8: Reading competed in the Championship in the 2016-17 season, finishing 3rd.
  Fact 0.9: Reading's website is listed as the club website.
Sentence 1: Reading Football Club ownership structure : 75 % Owned by Renhe Sports Management Ltd , 100 % owned by Xiu Li Dai and Yongge Dai . 25 % Owned by Narin Niruttinanon
  Fact 1.0: Reading Football Club's ownership structure is 75% owned by Renhe Sports Management Ltd.
  Fact 1.1: Renhe Sports Management Ltd is 100% owned by Xiu Li Dai and Yongge Dai.
  Fact 1.2: 25% of Reading Football Club is owned by Narin Niruttinanon.

Retrieved items:
<item 1>
Founded in 1871, Reading are among England’s oldest clubs and joined the Football League in 1920. They moved from Elm Park, their home for 102 years, to
Madejski Stadium in 1998. After first reaching the top flight in 2006–07, they returned in 2012–13 by winning the Championship but were relegated after one
season. Their records include 13 consecutive league wins to start 1985–86 and 106 points in 200
</item 1>

<item 2>
As of January 2018, Narin Niruttinanon owned 25% of Reading F.C. and was among its directors. The listed first-team squad included captain Paul McShane, Vito
Mannone, Liam Moore, John Swift, Garath McCleary, Sone Aluko and Yann Kermorgant. A 1999 supporters’ poll named Robin Friday “Player of the Millennium.” Senior
staff included chief executive Ron Gourlay; Jaap Stam managed the
</item 2>

<item 3>
Reading F.C.
25 % Owned by Narin Niruttinanon Reading F.C. Directors : Xiu Li Dai , Yongge Dai , Ronald Gourlay , Narin Niruttinanon , Nigel Howe Players ( edit ) As of 8
January 2018 First - team squad ( edit ) Note : Flags indicate national team as defined under FIFA eligibility rules . Players may hold more than one non-FIFA
nationality . No . Position Player GK Vito Mannone DF Chris Gunter MF Joey van den Berg 5 DF Paul McShane ( captain ) 6 DF Liam Moore 7 MF Roy Beerens 8 MF
George Evans 9 FW Joseph Mendes 10 MF John Swift 11 DF Jordan Obita 12 MF Garath McCleary 14 FW Sone Aluko 15 MF Callum Harriott 16 MF David Edwards 17 FW Modou
Barrow 18 FW Yann Kermorgant 19 MF Leandro Bacuna No .
</item 3>

<item 4>
Reading F.C.
This deal continued until the team entered administration in October 2014 . The sponsorship deal ended despite the team surviving and racing in the 2015 Formula
One season . During the 2015 -- 16 season , Waitrose was on the back of the home shirt whilst Euro Cake was on the back of the away shirt , meanwhile Legend
Alliance sponsored the shorts for the season . Waitrose and Euro Cake sponsored the home and away kits respectively again for the 2016 -- 17 season . Cherwell
Software took sponsorship of the back of the home shirt for the 2017 -- 18 season while Euro Cake sponsored the back of the away shirt again for the third
consecutive season . Ownership and finances ( edit ) Reading Football Club ownership structure : 75 % Owned by Renhe Sports Management Ltd , 100 % owned by Xiu
Li Dai and Yongge Dai . 25 % Owned by Narin Niruttinanon Reading F.C.
</item 4>

<item 5>
Reading F.C.
internationals Former players ( edit ) Main article : List of Reading F.C. players Club officials ( edit ) Board of Directors & Senior Club Staff Chief
Executive : Ron Gourlay Commercial Director : Adam Benson Club Secretary : Sue Hewett Financial Controller : Bryan Stabler Directors : Sir John Madejski ( Co
Chairman ) , Sasima Srivikorn ( Co Chairwoman ) , Nigel Howe , Narin Niruttinanon , Sumrith Thanakarnjanasuth , Taweesuk Srisumrid and Theekharoj Piamphongsarn
Management team Director of Football : Brian Tevreden First Team Manager : Jaap Stam Assistant Manager : Andries Ulderink & Said Bakkati First Team Coach :
Michael Gilkes Goalkeeping Coach : Dave Beasant First Team Sports Scientist : Tristan Baker First Team Physio : Jon Urwin Academy Manager : Lee Herron Assistant
Academy Manager : David Dodds Academy Loan Manager : Mikele Leigertwood Kit Manager : Stewart Bannister First Team Head Scout : Steve Shorey Masseur : Dan
Buchanan Managers ( edit ) Main article : List of Reading F.C.
</item 5>

<item 6>
Reading F.C.
On 24 May 2007 , it was announced that planning permission had been granted to extend the stadium to a capacity of 36,900 . Reading have made plans for a new
training ground at Bearwood Golf Club to replace Hogwood park their current training facility . On 5 July 2016 , at the end of Eamonn Dolan 's funeral , Reading
announced that the North Stand would be renamed the `` Eamonn Dolan Stand '' . Support ( edit ) See also : Supporters ' Trust at Reading Supporters at a Reading
match at Elm Park in 1913 In 2001 , Reading became the first football club to register their fans as an official member of their squad , giving the `` player ''
registered with squad number 13 as ' Reading Fans ' . Rivalries ( edit ) Before going out of business in 1992 , Aldershot were Reading 's biggest rivals . There
was a strong rift between the two sets of fans , with fighting between fans occurring on several occasions . Strong feelings remain between fans of Reading and
fans of Aldershot Town , the refounded club in Aldershot .
</item 6>

<item 7>
Reading Football Club is a professional club based in Reading, Berkshire, founded in 1871. Nicknamed “The Royals” because Berkshire is a royal county, they were
formerly called “The Biscuitmen.” Reading play at the 24,161-capacity Madejski Stadium, their home since 1998. In the 2016–17 season, they competed in the
Championship and finished third. The club was majority-owned by Dai Yongge and Dai Xiuli, chaired by Sir
</item 7>

<item 8>
Reading F.C. was founded on 25 December 1871 after a meeting organised by future secretary Joseph Edward Sydenham. The club finished eighth in its first Premier
League season, 2006–07. In 2016–17, under Jaap Stam, Reading finished third in the Championship, beat Fulham in the play-offs, then lost the final to
Huddersfield on penalties. The club’s blue-and-white crest features a crown and Maiwand Lion. Its grounds included
</item 8>

<item 9>
Reading F.C.
Reading FC - Wikipedia Reading FC Jump to : navigation , search Not to be confused with Reading F.C. Women . Reading Full name Reading Football Club Nickname (
s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir
John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season Reading Football
Club ( / ˈrɛdɪŋ / ( listen ) RED - ing ) is a professional association football club based in Reading , Berkshire , England . The team play in the Championship
, the second tier of English football . Reading are nicknamed The Royals , due to Reading 's location in the Royal County of Berkshire , though they were
previously known as The Biscuitmen , due to the town 's association with Huntley and Palmers .
</item 9>

<item 10>
Reading F.C.
Reading then finished eighth in the 2006 -- 07 Premier League , their first ever season as a top flight club . Contents ( hide ) 1 History 2 Crest and colours 3
Stadium 4 Support 4.1 Rivalries 5 Sponsorship 5.1 Additional kit sponsors 6 Ownership and finances 7 Players 7.1 First - team squad 7.2 Under - 23 squad 7.3
Under - 18 squad 7.4 Out on loan 8 Records and statistics 8.1 Notable players 8.2 Player of the season 8.3 International players 8.4 Former players 9 Club
officials 10 Managers 11 Women 's team 12 Honours 12.1 Managerial 13 Affiliated clubs 14 References 15 Bibliography 16 External links History ( edit ) Main
article : History of Reading F.C. See also : Thames Valley Royals proposal Reading were formed on 25 December 1871 , following a public meeting at the Bridge
Street Rooms organised by the future club secretary Joseph Edward Sydenham .
</item 10>
=== END ITEM id=6db45cd98c2a2ca0 ===

=== ITEM id=d15cffe084829a76 ===
Gold sentences and facts:
Sentence 0: The music video for `` Total Eclipse of the Heart '' was directed by Russell Mulcahy and was filmed on location at the Holloway Sanatorium , a large Victorian Gothic hospital near Virginia Water , Surrey , England . The video features Bonnie Tyler clad in white , dreaming or fantasizing about her students in a boys ' boarding school . Young men are seen dancing and participating in various school activities and singing in a choir .
  Fact 0.0: The music video for 'Total Eclipse of the Heart' was directed by Russell Mulcahy.
  Fact 0.1: The music video for 'Total Eclipse of the Heart' was filmed on location at the Holloway Sanatorium.
  Fact 0.2: The Holloway Sanatorium is a large Victorian Gothic hospital.
  Fact 0.3: The Holloway Sanatorium is near Virginia Water, Surrey, England.
  Fact 0.4: The music video features Bonnie Tyler.
  Fact 0.5: In the music video, Bonnie Tyler is clad in white.
  Fact 0.6: In the music video, Bonnie Tyler is dreaming or fantasizing about her students in a boys' boarding school.
  Fact 0.7: In the music video, young men are seen dancing.
  Fact 0.8: In the music video, young men are seen participating in various school activities.
  Fact 0.9: In the music video, young men are seen singing in a choir.

Retrieved items:
<item 1>
Total Eclipse of the Heart
; the singer who performed for this re-dub was Scott 's friend Felisha Noble using the pseudonym Persephone Maewyn . The video is still viewable on Funny or Die
's website , where it has received over 100,000 views , qualifying it for `` Immortal '' status since April 2012 . In 2010 , Tyler appeared in an advertisement
for MasterCard , performing a short parody of the song with its noted new lyric `` Turn around , Neville . '' She performed the original song in a similar
advertisement for Westpac in 2012 . Warby Parker produced a parody video in anticipation of the solar eclipse of August 21 , 2017 . See also ( edit ) 1983 in
British music List of best - selling singles List of Billboard Hot 100 number - one singles of 1983 List of number - one singles of 1983 ( Australia ) List of
number - one singles of 1983 ( Canada ) List of number - one singles of 1983 ( Ireland ) List of number - one singles in 1983 ( New Zealand ) References ( edit
) ^ Jump up to : `` Tyler releases new Total Eclipse '' .
</item 1>

<item 2>
Total Eclipse of the Heart
Steinman said in an interview with Playbill : with ' Total Eclipse of the Heart ' , I was trying to come up with a love song and I remembered I actually wrote
that to be a vampire love song . Its original title was ' Vampires in Love ' because I was working on a musical of Nosferatu , the other great vampire story .
If anyone listens to the lyrics , they 're really like vampire lines . It 's all about the darkness , the power of darkness and love 's place in the dark ... He
also told People magazine that he thought Tyler sounded like John Fogerty , and wrote the song `` to be a showpiece for her voice . '' Tyler described the song
as `` a challenge ( to sing ) , '' stating that she `` ( does n't ) like songs that anybody can sing . I like songs that need a lot of energy . '' After
Steinman presented her with the song she told The Times , `` I just had shivers right up my spine ... I could n't wait to actually get in and record it . ''
</item 2>

<item 3>
Total Eclipse of the Heart
( Sunset Strippers Verse Club Mix ) `` Total Eclipse of the Heart '' ( Sunset Strippers Dub Mix ) `` Total Eclipse of the Heart '' ( Sunset Strippers Radio Edit
) Cover versions and impact in Mainstream Media ( edit ) In a 2013 UK survey , the song came first in a list of most popular songs to sing in the shower , above
songs by Justin Bieber , Robbie Williams , One Direction and Elton John . In 2015 the song was voted by the British public as the nation 's third favourite
1980s number one in a poll for ITV . `` Total Eclipse of the Heart '' received substantial media attention during the solar eclipse of March 20 , 2015 . Tyler
's version received a 214 % increase of Spotify streams throughout the day . A similar impact was experienced during the solar eclipse of August 21 , 2017 ,
when Nielsen Music reported a 503 % increase in record sales . Around that time , the song hit number one on the iTunes chart .
</item 3>

<item 4>
Total Eclipse of the Heart
230 . Jump up ^ Scott , David ( 16 April 2011 ) . `` Total Eclipse of the Heart : Literal Video Version '' . Retrieved 26 August 2017 . Jump up ^ Benedictus ,
Leo ( 21 April 2010 ) . `` A word on our sponsors : Total eclipse of the card '' . The Guardian . Retrieved 20 August 2014 . Jump up ^ `` Westpac brings in
Bonnie Tyler to sing at couple 's fantasy wedding '' . mUmBRELLA. 21 November 2012 . Retrieved 22 August 2014 . Jump up ^ Matthewson , Samantha ( 1 August 2017
) . `` This ' Total Eclipse of the Heart ' Parody Video for the 2017 Solar Eclipse Is Just Awesome '' . Space.com . Retrieved 3 August 2017 .
</item 4>

<item 5>
Total Eclipse of the Heart
The song would have been released as the album 's second single but was cancelled due to their The Love Tour conflicts , though a promo release still reached
No. 5 on the radio charts in the Philippines . Three official remixes have been made for their version as well as a remix done by Jim Steinman , which was
ultimately rejected by the record label but has surfaced on the Internet . The song was released as promotional single on 2007 . Yuridia version ( edit )
Mexican singer Yuridia covered the song in Spanish as `` Eclipse Total del Amor '' from her second studio album Habla El Corazón in 2006 . The single peaked at
number 36 on the Billboard Latin Pop Songs chart . Parody versions ( edit ) A parody of the song and music video were published in 2009 , in what the fans and
makers call a `` literal video version '' , which is a type of video that replaces the original song lyrics with humorous lyrics describing the images in the
video . Time magazine listed it as the 6th best viral video of 2009 . This was also the 6th literal video produced by professional video editor David A. Scott
Jr.
</item 5>

<item 6>
Total Eclipse of the Heart
) `` Sayonara Tokyo '' ( 1981 ) `` Total Eclipse of the Heart '' ( 1983 ) `` Take Me Back '' ( 1983 ) Music video `` Total Eclipse Of The Heart '' on YouTube ``
Total Eclipse of the Heart '' is a song recorded by Welsh singer Bonnie Tyler . It was written and produced by Jim Steinman , and released on Tyler 's fifth
studio album , Faster Than the Speed of Night ( 1983 ) . The song was released as a single by Columbia Records on 11 February 1983 in the United Kingdom and on
31 May 1983 in the United States . The song became Tyler 's biggest career hit , topping the UK Singles Chart , and becoming the fifth - best - selling single
in 1983 in the United Kingdom . In the United States , the single spent four weeks at the top of the charts , and was Billboard 's number - six song of the year
for 1983 .
</item 6>

<item 7>
Total Eclipse of the Heart
Total Eclipse of the Heart - wikipedia Total Eclipse of the Heart Jump to : navigation , search For the episode of `` The Vampire Diaries '' , see Total Eclipse
of the Heart ( The Vampire Diaries ) . `` Total Eclipse of the Heart '' Artwork for UK and some European releases Single by Bonnie Tyler from the album Faster
Than the Speed of Night B - side `` Take Me Back '' ( UK ) `` Straight from the Heart '' ( US ) Released Format 7 '' 12 '' CD Length 6 : 59 ( album version ) 5
: 53 ( 2013 re-recording for the album Rocks and Honey ) 5 : 32 ( music video version ) 4 : 35 ( 2011 single version ) 4 : 30 ( single version ) 3 : 49 ( 2005
re-recording for the album Wings ) Label Columbia Songwriter ( s ) Jim Steinman Producer ( s ) Jim Steinman Bonnie Tyler singles chronology `` Sayonara Tokyo ''
( 1981 ) `` Total Eclipse of the Heart '' ( 1983 ) `` Take Me Back '' ( 1983 ) `` Sayonara Tokyo '' ( 1981 ) `` Total Eclipse of the Heart
</item 7>

<item 8>
Total Eclipse of the Heart
Donald A. Guarisco , also from AllMusic , retrospectively reviewed Faster Than the Speed of Night , and noted the song as an `` epic ballad , '' describing the
whole album as `` rock at its most melodramatic . '' Jim Beviglia from American Songwriter said that Tyler 's raspy vocals helped to legitimize the `` melodrama
inherent in the lyrics , '' and described the song as a `` garment - rending , chest - beating ( and ) emotionally exhausting ballad '' that suits the throes of
a turbulent relationship . Music video ( edit ) Holloway Sanatorium steps , which feature in the video The music video for `` Total Eclipse of the Heart '' was
directed by Russell Mulcahy and was filmed on location at the Holloway Sanatorium , a large Victorian Gothic hospital near Virginia Water , Surrey , England .
The video features Bonnie Tyler clad in white , dreaming or fantasizing about her students in a boys ' boarding school . Young men are seen dancing and
participating in various school activities and singing in a choir .
</item 8>

<item 9>
Total Eclipse of the Heart
Young men are seen dancing and participating in various school activities and singing in a choir . A long - running urban legend is that the boy who appears
throughout the video and who shakes Tyler 's hand at the end is former Italian footballer Gianfranco Zola . In a 2012 interview , Zola confirmed that he did not
appear in the video . Live performances ( edit ) Since the song 's release , Tyler performs `` Total Eclipse of the Heart '' in every concert she performs . ``
I sing it much better now than I used to , '' she told The Huffington Post . `` I think my voice is probably not as husky as it was , I think it 's mellowed a
bit . '' The song was performed at the 26th Annual Grammy Awards , held at the Shrine Auditorium , Los Angeles , on 28 February 1984 . Tyler performed a brief ,
live performance of this song on a Royal Caribbean cruise during the solar eclipse of 21 August 2017 . Live recordings of Tyler performing the song have been
released on her live albums Bonnie Tyler Live ( 2006 ) and Live in Germany 1993 ( 2011 ) .
</item 9>

<item 10>
Total Eclipse of the Heart
He also hand - picked the recording band for the song . `` Total Eclipse of the Heart '' had to be shortened for radio play . Tyler did not believe that the
song was radio - friendly at its full length ; the song was reduced from seven minutes and two seconds to four minutes and thirty seconds . The power ballad
remains Tyler 's most successful song , peaking at No. 1 in the United States , South Africa , Australia , New Zealand , Canada and the United Kingdom . At its
peak , it sold 60,000 copies per day , and approximately 6 million copies in total . It won the Variety Club award in the UK for best single of 1983 . The song
also made number 82 of VH1 's top 100 love songs . `` Total Eclipse of the Heart '' A sample of `` Total Eclipse of the Heart '' , performed by Bonnie Tyler .
Problems playing this file ? See media help . Tyler told Record Mirror that she thought the song was about `` someone who wants to love so badly she 's lying
there in complete darkness . ''
</item 10>
=== END ITEM id=d15cffe084829a76 ===

=== ITEM id=481d9f0528e6bca8 ===
Gold sentences and facts:
Sentence 0: A '' Time for Choosing Speech '' Play media In support of Goldwater , Reagan delivers the address , A Time for Choosing . The speech launches Reagan to national prominence . Date October 27 , 1964 ( 1964 - 10 - 27 ) Duration 29 : 33 Location Los Angeles County , California , U.S. Also known as `` The Speech '' Type Televised campaign speech Participants Ronald Reagan Website Video clip , audio , transcript
  Fact 0.0: The speech 'A Time for Choosing' is also known as 'The Speech'.
  Fact 0.1: The speech 'A Time for Choosing' is a televised campaign speech.
  Fact 0.2: Ronald Reagan delivers the address 'A Time for Choosing' in support of Barry Goldwater.
  Fact 0.3: The speech launches Reagan to national prominence.
  Fact 0.4: The speech is dated October 27, 1964.
  Fact 0.5: The speech runs 29 minutes and 33 seconds.
  Fact 0.6: The speech was held in Los Angeles County, California, U.S.
  Fact 0.7: The participant in the speech is Ronald Reagan.
  Fact 0.8: A video clip, audio, and a transcript of the speech are available on a website.
Sentence 1: Many versions of the speech exist , since it was altered over many weeks . Contrary to popular belief , however , the speech was not given at the 1964 Republican National Convention in San Francisco , California as a nomination speech for presidential candidate Senator Barry Goldwater ; Richard Nixon gave that nomination speech . Reagan , though he campaigned for Goldwater , did not use `` A Time for Choosing '' until October 27 , 1964 , when it was part of a pre-recorded television program , Rendezvous with Destiny . In his autobiography , An American Life , Reagan recalled going to bed that night `` hoping I had n't let Barry down . ''
  Fact 1.0: Many versions of the speech exist.
  Fact 1.1: The speech was altered over many weeks.
  Fact 1.2: Contrary to popular belief, the speech was not given at the 1964 Republican National Convention in San Francisco, California, as a nomination speech for presidential candidate Barry Goldwater.
  Fact 1.3: Richard Nixon gave the nomination speech for Goldwater at the 1964 Republican National Convention.
  Fact 1.4: Reagan campaigned for Goldwater.
  Fact 1.5: Reagan did not use 'A Time for Choosing' until October 27, 1964.
  Fact 1.6: On October 27, 1964, the speech was part of a pre-recorded television program called Rendezvous with Destiny.
  Fact 1.7: In his autobiography, An American Life, Reagan recalled going to bed that night hoping he had not let Barry down.

Retrieved items:
<item 1>
A Time for Choosing
In his autobiography , An American Life , Reagan recalled going to bed that night `` hoping I had n't let Barry down . '' Speaking for Goldwater , Reagan
stressed his belief in the importance of smaller government . In the speech , he revealed his ideological motivation : `` The Founding Fathers knew a government
ca n't control the economy without controlling people . And they knew when a government sets out to do that , it must use force and coercion to achieve its
purpose . So we have come to a time for choosing . '' He also said , `` You and I are told we must choose between a left or right , but I suggest there is no
such thing as a left or right . There is only an up or down . Up to man 's age - old dream -- the maximum of individual freedom consistent with law and order --
or down to the ant heap of totalitarianism . '' The speech raised $1 million for Goldwater 's campaign , and is considered the event that launched Reagan 's
political career . Aftermath ( edit ) To this day , `` The Speech '' is considered one of the most effective ever made on behalf of a candidate .
</item 1>

<item 2>
A Time for Choosing
Family Jack Reagan ( father ) Nelle Wilson Reagan ( mother ) Neil Reagan ( brother ) Jane Wyman ( first wife ) Nancy Reagan ( second wife ) Maureen Reagan (
daughter ) Michael Reagan ( adopted son ) Patti Davis ( daughter ) Ron Reagan ( son ) Rex ( dog ) ← Jimmy Carter George H.W. Bush → Book Retrieved from ``
https://en.wikipedia.org/w/index.php?title=A_Time_for_Choosing&oldid=805876525 '' Categories : Speeches by Ronald Reagan 1964 speeches United States
presidential election , 1964 Hidden categories : All articles with unsourced statements Articles with unsourced statements from March 2016 Articles containing
video clips Talk Contents About Wikipedia Deutsch עברית Português Scots Simple English Edit links This page was last edited on 18 October 2017 , at 05 : 38 .
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
A Time for Choosing
Santa Barbara , California : ABC - CLIO , 2009 . p. 384 Jump up ^ Cannon , Lou ( June 6 , 2004 ) . `` Actor , Governor , President , Icon '' . The Washington
Post . p . A01 . Retrieved January 26 , 2008 . Jump up ^ Cannon ( 2001 ) , p. 36 Jump up ^ Peter Schweizer ; Wynton C. Hall ( 2007 ) . Landmark Speeches of the
American Conservative Movement . Texas A&M University Press . p. 42 . Retrieved April 26 , 2012 . Further reading ( edit ) Bimes , Terri . `` Ronald Reagan and
the New Conservative Populism . '' ( Institute of Governmental Studies , 2002 ) . online Friedenberg , Robert V. ( 2002 ) . Notable Speeches in Contemporary
Presidential Campaigns . Greenwood . pp. 143ff . Kiewe , Amos , Davis W. Houck , and Davis L. Houck .
</item 3>

<item 4>
A Time for Choosing
A Time for Choosing - wikipedia A Time for Choosing Jump to : navigation , search A '' Time for Choosing Speech '' Play media In support of Goldwater , Reagan
delivers the address , A Time for Choosing . The speech launches Reagan to national prominence . Date October 27 , 1964 ( 1964 - 10 - 27 ) Duration 29 : 33
Location Los Angeles County , California , U.S. Also known as `` The Speech '' Type Televised campaign speech Participants Ronald Reagan Website Video clip ,
audio , transcript This article is part of a series about Ronald Reagan Electoral history Political positions Early life Career Films Public image Governor of
California Governorship 1976 General election Primaries Convention 40th President of the United States Presidency Timeline Policies Domestic Reaganomics Foreign
Reagan Doctrine International trips Appointments Cabinet Judicial appointments First Term Campaign for the Presidency 1980 general election Primaries Convention
1st inauguration Assassination attempt Invasion of Grenada Cold War Second Term Re-election campaign 1984 general election Primaries Convention 2nd inauguration
Cold War Libya bombing Challenger disaster Iran -- Contra affair `` Tear down this wall ! ''
</item 4>

<item 5>
A Time for Choosing
audio , and video of the broadcast Analysis of `` The Speech '' , The History Channel ( hide ) Ronald Reagan 40th President of the United States ( 1981 -- 1989
) 33rd Governor of California ( 1967 -- 1975 ) Life and politics Birthplace Pitney Store Boyhood home Rancho del Cielo Filmography Presidential Library Death
and state funeral Political positions Governship of California Namesakes and memorials Presidency First inauguration Second inauguration Domestic policy
Economic policy Economic Recovery Tax Act of 1981 Tax Reform Act of 1986 Assassination attempt Strategic Defense Initiative Foreign policy Reagan Doctrine Cold
War 1st term 2nd term 1985 Geneva Summit 1986 Reykjavík Summit INF Treaty 1987 Washington Summit 1988 Moscow Summit Invasion of Grenada Iran -- Contra affair
International trips The Grace Commission Cabinet Federal judicial appointments Supreme Court controversies Administration scandals `` We begin bombing in five
minutes '' Speeches Ronald Reagan Speaks Out Against Socialized Medicine `` A Time for Choosing '' Reagan 's Neshoba County Fair `` states ' rights '' speech
First inaugural address Second inaugural address `` Ash heap of history '' `` Evil empire '' `` Tear down this wall ! ''
</item 5>

<item 6>
A Time for Choosing
INF Treaty Post-Presidency Presidential Library Medal of Freedom Bibliography An American Life The Reagan Diaries Alzheimer 's diagnosis State funeral Legacy
Speeches and debates `` A Time for Choosing '' Reagan Era Reagan Award `` A Time for Choosing '' , also known as `` The Speech '' , was a speech presented
during the 1964 U.S. presidential election campaign by future president Ronald Reagan on behalf of Republican candidate Barry Goldwater . The speech launched
Reagan into national prominence . Contents ( hide ) 1 Background 2 Aftermath 3 References 4 Further reading 5 External links Background ( edit ) Many versions
of the speech exist , since it was altered over many weeks . Contrary to popular belief , however , the speech was not given at the 1964 Republican National
Convention in San Francisco , California as a nomination speech for presidential candidate Senator Barry Goldwater ; Richard Nixon gave that nomination speech .
Reagan , though he campaigned for Goldwater , did not use `` A Time for Choosing '' until October 27 , 1964 , when it was part of a pre-recorded television
program , Rendezvous with Destiny .
</item 6>

<item 7>
A Time for Choosing
A shining city on a hill : Ronald Reagan 's economic rhetoric , 1951 -- 1989 ( Greenwood , 1991 ) Perlstein , Rick ( 2009 ) . Before the Storm : Barry Goldwater
and the Unmaking of the American Consensus . Nation Books . pp. 509ff . Ritter , Kurt W. `` Ronald Reagan and ' the speech ' : The rhetoric of public relations
politics . '' Western Journal of Communication ( 1968 ) 32 # 1 pp. 50 -- 58 External links ( edit ) Wikisource has original text related to this article : A
Time for Choosing Ronald Reagan , `` A time for choosing '' ( 1964 ) Online Transcript , audio , and video of the broadcast Analysis of `` The Speech '' ,
</item 7>

<item 8>
Gerald Ford
Jump up ^ Shabecoff , Philip . `` 160,000 Mark Two 1775 Battles ; Concord Protesters Jeer Ford -- Reconciliation Plea '' , The New York Times , April 20 , 1975
, p. 1 . Jump up ^ Shabecoff , Philip . `` Ford , on Bicentennial Trip , Bids U.S. Heed Old Values '' , The New York Times , April 19 , 1975 , p. 1 . Jump up ^
`` 1976 Presidential Debates '' . CNN . Retrieved September 28 , 2011 . Jump up ^ Lehrer , Jim ( 2000 ) . `` 1976 : No Audio and No Soviet Domination '' .
Debating Our Destiny . PBS . Retrieved March 31 , 2007 . Jump up ^ `` Presidential Election 1976 States Carried '' . multied.com . Retrieved December 31 , 2006
. Jump up ^ `` Jimmy Carter '' . U.S. Inaugural Addresses . Bartleby.com . January 20 , 1977 .
</item 8>

<item 9>
A Time for Choosing
Nevertheless , Barry Goldwater lost the election by one of the largest margins in history . Soon afterwards , Reagan was asked to run for Governor of California
; he ran for office and won election in 1966 . Reagan was later dubbed the `` Great Communicator '' in recognition of his effective oratory skills . References
( edit ) Jump up ^ Reagan , Ronald ( 1990 ) . An American Life : The Autobiography . New York City : Simon & Schuster . p. 143 . ISBN 0671691988 . Retrieved 31
March 2014 . When it was over , the others in the room said I had done well . But I was still nervous about it and , when I went to bed , I was hoping I had n't
let Barry down ^ Jump up to : `` A Time for Choosing '' . PBS . Retrieved April 17 , 2007 . Jump up ^ Richard A. Harris , Daniel J. Tichenor . A History of the
U.S. Political System : Ideas , Interests , and Institutions , Volume 1 . Santa Barbara , California : ABC - CLIO , 2009 .
</item 9>

<item 10>
A Time for Choosing
State of the Union : 1982 1984 1985 1986 Books Where 's the Rest of Me ? ( 1965 autobiography ) ( with Richard G. Hubler ) An American Life ( 1990 autobiography
) ( with Robert Lindsey ) The Reagan Diaries ( 2007 ) ( edited by Douglas Brinkley ) Elections California gubernatorial election , 1966 1970 Republican Party
presidential primaries , 1968 1976 1980 1984 Republican National Convention 1968 1976 1980 1984 Ronald Reagan presidential campaign , 1980 `` There you go again
'' `` Make America Great Again '' United States presidential election , 1976 1980 1984 `` Morning in America '' `` Bear in the woods '' Popular culture In
fiction In music U.S. Postage stamps The Day Reagan Was Shot ( 2001 film ) The Reagans ( 2003 film ) Reagan ( 2011 documentary ) The Butler ( 2013 film )
Killing Reagan ( 2016 film ) `` What would Reagan do ? ''
</item 10>
=== END ITEM id=481d9f0528e6bca8 ===
