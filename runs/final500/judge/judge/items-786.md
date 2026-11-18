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

=== ITEM id=824a3e2b589c52de ===
Gold sentences and facts:
Sentence 0: A third series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be set in 1937 .
  Fact 0.0: A third series was confirmed to be in production by writer Simon Nye.
  Fact 0.1: Simon Nye confirmed the third series at the BFI and Radio Times Television Festival on 8 April 2017.
  Fact 0.2: Simon Nye described the third series as having some exotic new animals.
  Fact 0.3: Simon Nye said production of the third series would begin in three weeks' time.
  Fact 0.4: Production of the third series would begin upon Keeley Hawes's arrival in Corfu for filming.
  Fact 0.5: The third series will be set in 1937.

Retrieved items:
<item 1>
The Durrells
5 5 `` Episode 5 '' Roger Goldby Simon Nye 1 May 2016 ( 2016 - 05 - 01 ) 6.93 A visit from some priggish relatives seems to deem the family 's immigration to
Corfu more temporary than first thought , and Louisa 's aunt 's insistence that they must return to England is made even more demanding by the revelation of her
relationship with Sven . Larry , despondent at the rejection of his novel by publishers , subdues himself into a life of labour -- soon discovering his
ineptness . Margo tries to balance her devotion to the Countess and her growing adoration for her gardener , and Gerry 's plans to embark into the world of
conservation are thrown into disarray by his attempts to feed his adopted baby owls . 6 6 `` Episode 6 '' Roger Goldby Simon Nye 8 May 2016 ( 2016 - 05 - 08 )
6.58 The family 's concerns over Sven 's true intentions are brought to light when Louisa realises that Sven 's romantic feelings for her are merely a ruse to
hide his homosexuality .
</item 1>

<item 2>
The Durrells
Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 - 04 - 03 ) 8 May 2016 ( 2016 - 05 - 08 ) 6 23 April 2017 ( 2017 - 04 - 23 ) 28 May 2017 (
2017 - 05 - 28 ) Episodes ( edit ) Series 1 ( 2016 ) ( edit ) ( 1935 ) No . overall No. in series Title Directed by Written by Original air date UK viewers (
millions ) `` Episode 1 '' Steve Barron Simon Nye 3 April 2016 ( 2016 - 04 - 03 ) 8.81 The island of Corfu 's distinct contrast with England is what enthrals
Bournemouth mother Louisa Durrell to uproot her disenchanted family from a life of no prospects and venture into one with a lessened need for conformity .
However , the appeal becomes ever distant to see upon their arrival .
</item 2>

<item 3>
The Durrells
The show and its episodes attained high ratings on IMDb , around 8 / 10 . BROADCAST ( edit ) Internationally , the series was acquired in Australia by the Seven
Network and premiered on 24 August 2016 . In the United States , PBS began airing the show , retitled as The Durrells in Corfu , on 16 October 2016 at 8pm . The
Durrells started screening in New Zealand on 26 October 2016 on Prime TV . In Canada , the Canadian Broadcasting Corporation began airing Series 1 on 13
September 2017 . In Spain , the series has been acquired by the streaming platform Filmin . A third series of The Durrells was announced by writer Simon Nye at
the BFI and Radio Times Television Festival on 8 April 2017 . It began filming in May 2017 and is expected to air on ITV in the Spring of 2018 .
</item 3>

<item 4>
The Durrells
The Durrells - wikipedia The Durrells Jump to : navigation , search The Durrells Genre Comedy - drama Based on The Corfu Trilogy by Gerald Durrell Written by
Simon Nye Theme music composer Ruth Barrett Country of origin United Kingdom Original language ( s ) English No. of series No. of episodes 12 Production
Executive producer ( s ) Lee Morris Sally Woodward Gentle Simon Nye Producer ( s ) Christopher Hall Cinematography Julian Court , James Aspinall Running time 46
-- 47 minutes Production company ( s ) Sid Gentle Films Masterpiece co-production Release Original network ITV Picture format 16 : 9 1080i Audio format Stereo
Original release 3 April 2016 ( 2016 - 04 - 03 ) -- present External links Website The Durrells ( also known as The Durrells in Corfu on American television )
is a British comedy - drama series based on Gerald Durrell 's three autobiographical books about his family 's four years ( 1935 -- 1939 ) on the Greek Island
of Corfu , which began airing on 3 April 2016 .
</item 4>

<item 5>
The Durrells
Characters ( edit ) Main Keeley Hawes as Louisa Durrell Milo Parker as Gerry Durrell Josh O'Connor as Larry Durrell Daisy Waterstone as Margo Durrell Callum
Woodhouse as Leslie Durrell Lucy Black as Florence Petrides Yorgos Karamihos as Dr. Theo Stephanides Alexis Georgoulis as Spiros Halikiopoulos Leslie Caron as
Countess Mavrodaki Ulric von der Esch as Sven Lundblad Anna Savva as Lugaretzia Errika Bigiou as Vasilia ( series 2 ) Recurring James Cosmo as Captain Creech
Maximilian Befort as Max Manolis Emmanouel as Sotos Andrew Bicknell as Headmaster Graham Seed as Mr Trevitt Liz Watts as Nancy Ben Hall as Donald Hara - Joy
Ermidi as Alexia Yorgos Tryfonas as Market Trader Nick Orestis Chaniotakis as Monk Arsenis Grimmas as Guest House Manager Spiros Kasfikis as Durrell 's House
Owner Daniel Lapaine as Hugh Jarvis Series overview ( edit ) Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 -
</item 5>

<item 6>
The Durrells
12 6 `` Episode 6 '' Edward Hall Simon Nye 28 May 2017 ( 2017 - 05 - 28 ) 5.57 Three different children are born as a fraught Leslie is on hand to help
Lugaretzia 's daughter give birth , Louisa aids with Florence 's , and Gerry is overwhelmed with exuberance at the prospect of some otter cubs - all of which ,
as well as Corfu 's rich culture and wildlife , helps Louisa realise she should refuse to fulfill Hugh 's wish for the two of them to return to England . Hugh ,
however , finds himself in a more dangerous situation , as Vasilia , in a last - ditch attempt to win him back , and in a frantic displeasing at his choosing of
Louisa , stabs him , and flees ; Larry turns up at her doorstep for one last liaison , and she rejects him , adding more disappointment to his day as he later
discovers he has missed everything exciting , an opinion soon put to rest when Donald , prior to leaving Corfu , engages in a fight with Margo 's conceited new
lover , Zoltan .
</item 6>

<item 7>
The Durrells
Jump up ^ The episode 's world premiere date was at the BFI and Radio Times Television Festival on 8 April 2017 . Series 3 ( 2018 ) ( edit ) ( 1937 ) A third
series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as
having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be
set in 1937 . Reception ( edit ) Reception to the first episode was positive , with Gerard O'Donovan ( The Telegraph ) calling it `` a series that 's not only
sun - drenched and liberating , but also catches its source material 's high good humour without labouring it and weaves an authentic sense of the innocent
exoticism of the original , '' before awarding it four stars .
</item 7>

<item 8>
The Durrells
The series is written by Simon Nye , directed by Steve Barron and Roger Goldby , and produced by Christopher Hall . The executive producers are Lee Morris and
Sally Woodward Gentle. Lee Durrell , Gerald Durrell 's widow and director of the Durrell Wildlife Conservation Trust , acted as consultant . The series is
partly filmed on Corfu and at Ealing Studios in London . Contents ( hide ) 1 Story 2 Characters 3 Series overview 4 Episodes 4.1 Series 1 ( 2016 ) 4.2 Series 2
( 2017 ) 4.3 Series 3 ( 2018 ) 5 Reception 6 Broadcast 7 See also 8 References 9 External links Story ( edit ) The series begins in 1935 , when Louisa Durrell
suddenly announces that she and her four children will move from Bournemouth to the Greek island of Corfu . Her husband has died some years earlier and the
family is experiencing financial problems . A Homeric battle ensues as the family adapts to life on the island which , despite a lack of electricity , is cheap
and an earthly paradise .
</item 8>

<item 9>
The Durrells
See also ( edit ) Gerald Durrell 's Corfu trilogy books : My Family and Other Animals ( 1956 ) Birds , Beasts , and Relatives ( 1969 ) The Garden of the Gods (
1978 ) My Family and Other Animals , a 1987 BBC series based on the Corfu trilogy by Durrell My Family and Other Animals , a 2005 BBC telemovie based on the
1956 book by Durrell References ( edit ) Jump up ^ `` Early life of Gerald Durrell comes to TV screens soon '' . Jersey Evening Post . 24 March 2016 . Retrieved
1 April 2016 . Jump up ^ `` ITV commissions new six - part drama The Durrells '' . ITV Press Centre . 18 June 2015 . Retrieved 1 April 2016 . Jump up ^ `` The
Durrells ( TV Series 2016 -- ) '' . Internet Movie Database . Retrieved 1 April 2016 . Jump up ^ Huntman , Ruth ( 26 March 2016 ) .
</item 9>

<item 10>
The Durrells
Jump up ^ `` Twitter - ITV Media '' . Twitter . Retrieved 11 April 2016 . Jump up ^ `` Digital Spy - Forums - The Ratings Thread ( Part 67 ) '' . Digital Spy
Forums . Retrieved 11 April 2016 . Jump up ^ `` BARB - Weekly Top 10 Programmes ( w / e 3rd April 2016 ) '' . Broadcasters ' Audience Research Board . Retrieved
11 April 2016 . Jump up ^ `` ITV Media - Press Releases - `` ITV recommissions hit drama The Durrells '' `` . ITV Media . Retrieved 16 April 2016 . Jump up ^ ``
ITV Press Centre - `` More Sun , Sea and Animals as the cast of The Durrells return to Corfu to film the second series of the ITV drama '' `` . ITV Press Centre
. ITV . Retrieved 12 August 2016 . Jump up ^ `` IMDb - The Durrells '' . Internet Movie Database . Retrieved 12 August 2016 .
</item 10>
=== END ITEM id=824a3e2b589c52de ===

=== ITEM id=aa0b8fd2d63d213d ===
Gold sentences and facts:
Sentence 0: Driver Seasons Entries Podiums Percentage Michael Schumacher 1991 -- 2006 , 2010 -- 2012 308 155 7001503200000000000 ♠ 50.32 % Lewis Hamilton 2007 -- 2018 211 119 7001564000000000000 ♠ 56.40 % Alain Prost 1980 -- 1991 , 1993 202 106 7001524800000099999 ♠ 52.48 % Sebastian Vettel 2007 -- 2018 202 101 7001500000000000000 ♠ 50.00 % 5 Fernando Alonso 2001 , 2003 -- 2018 296 97 7001327700000000000 ♠ 32.77 % 6 Kimi Räikkönen 2001 -- 2009 , 2012 -- 2018 276 93 7001337000000000000 ♠ 33.70 % 7 Ayrton Senna 1984 -- 1994 162 80 7001493800000000000 ♠ 49.38 % 8 Rubens Barrichello 1993 -- 2011 326 68 7001208600000000000 ♠ 20.86 % 9 David Coulthard 1994 -- 2008 247 62 7001251000000000000 ♠ 25.10 % 10 Nelson Piquet 1978 -- 1991 207 60 7001289900000000000 ♠ 28.99 % Source :
  Fact 0.0: The table lists drivers with their seasons, entries, podiums, and percentage.
  Fact 0.1: Michael Schumacher raced in the seasons 1991 to 2006 and 2010 to 2012.
  Fact 0.2: Michael Schumacher had 308 entries.
  Fact 0.3: Michael Schumacher had 155 podiums.
  Fact 0.4: Michael Schumacher has a percentage of 50.32%.
  Fact 0.5: Lewis Hamilton raced in the seasons 2007 to 2018.
  Fact 0.6: Lewis Hamilton had 211 entries.
  Fact 0.7: Lewis Hamilton had 119 podiums.
  Fact 0.8: Lewis Hamilton has a percentage of 56.40%.
  Fact 0.9: Alain Prost raced in the seasons 1980 to 1991 and 1993.
  Fact 0.10: Alain Prost had 202 entries.
  Fact 0.11: Alain Prost had 106 podiums.
  Fact 0.12: Alain Prost has a percentage of 52.48%.
  Fact 0.13: Sebastian Vettel raced in the seasons 2007 to 2018.
  Fact 0.14: Sebastian Vettel had 202 entries.
  Fact 0.15: Sebastian Vettel had 101 podiums.
  Fact 0.16: Sebastian Vettel has a percentage of 50.00%.
  Fact 0.17: Fernando Alonso is ranked 5th in the table.
  Fact 0.18: Fernando Alonso raced in the seasons 2001 and 2003 to 2018.
  Fact 0.19: Fernando Alonso had 296 entries.
  Fact 0.20: Fernando Alonso had 97 podiums.
  Fact 0.21: Fernando Alonso has a percentage of 32.77%.
  Fact 0.22: Kimi Räikkönen is ranked 6th in the table.
  Fact 0.23: Kimi Räikkönen raced in the seasons 2001 to 2009 and 2012 to 2018.
  Fact 0.24: Kimi Räikkönen had 276 entries.
  Fact 0.25: Kimi Räikkönen had 93 podiums.
  Fact 0.26: Kimi Räikkönen has a percentage of 33.70%.
  Fact 0.27: Ayrton Senna is ranked 7th in the table.
  Fact 0.28: Ayrton Senna raced in the seasons 1984 to 1994.
  Fact 0.29: Ayrton Senna had 162 entries.
  Fact 0.30: Ayrton Senna had 80 podiums.
  Fact 0.31: Ayrton Senna has a percentage of 49.38%.
  Fact 0.32: Rubens Barrichello is ranked 8th in the table.
  Fact 0.33: Rubens Barrichello raced in the seasons 1993 to 2011.
  Fact 0.34: Rubens Barrichello had 326 entries.
  Fact 0.35: Rubens Barrichello had 68 podiums.
  Fact 0.36: Rubens Barrichello has a percentage of 20.86%.
  Fact 0.37: David Coulthard is ranked 9th in the table.
  Fact 0.38: David Coulthard raced in the seasons 1994 to 2008.
  Fact 0.39: David Coulthard had 247 entries.
  Fact 0.40: David Coulthard had 62 podiums.
  Fact 0.41: David Coulthard has a percentage of 25.10%.
  Fact 0.42: Nelson Piquet is ranked 10th in the table.
  Fact 0.43: Nelson Piquet raced in the seasons 1978 to 1991.
  Fact 0.44: Nelson Piquet had 207 entries.
  Fact 0.45: Nelson Piquet had 60 podiums.
  Fact 0.46: Nelson Piquet has a percentage of 28.99%.
  Fact 0.47: The table ends with a 'Source' label that has no source given.

Retrieved items:
<item 1>
List of Formula One driver records
ESPNF1 . Retrieved 9 January 2014 . Jump up ^ `` Most grand prix without pole '' . StatsF1.com . Retrieved 31 July 2017 . ^ Jump up to : `` Statistics Drivers
-- Fastests laps -- By age '' . StatsF1.com . Retrieved 2018 - 04 - 09 . ^ Jump up to : `` Podiums by number '' . StatsF1.com. 9 April 2017 . Retrieved 10 April
2017 . Jump up ^ `` Most consecutive podium finishes '' . StatsF1.com . Retrieved 10 May 2015 . Jump up ^ `` Grands Prix -- without podium '' . StatsF1.com. 23
July 2016 . Retrieved 23 July 2016 . Jump up ^ Collantine , Keith ( 11 April 2011 ) . `` Heidfeld sets record for most podiums without a win '' . F1 Fanatic .
Retrieved 10 January 2014 . Jump up ^ `` Top 100 F1 Drivers '' .
</item 1>

<item 2>
List of Formula One driver records
2009 -- 2011 81 80 6th 10 Marcus Ericsson 2014 -- 2018 79 79 8th Source : Most career podiums without a win ( edit ) Driver Entries Starts Podiums Nick Heidfeld
185 183 13 Stefan Johansson 103 79 12 Chris Amon 108 97 11 Romain Grosjean 127 125 10 5 Martin Brundle 165 158 9 Eddie Cheever 143 132 Jean Behra 53 52 8 Luigi
Villoresi 34 31 8 9 Sergio Pérez 140 137 7 10 Andrea de Cesaris 214 208 5 Source : Most career podiums without a World Championship ( edit ) Driver Seasons
Starts Wins Podiums Rubens Barrichello 1993 -- 2011 326 11 68 David Coulthard 1994 -- 2008 247 13 62 Gerhard Berger 1984 -- 1997 210 10 48 Carlos Reutemann 1972
-- 1982 146 12 45 5 Mark Webber 2002
</item 2>

<item 3>
List of Formula One driver records
43 days 1959 United States Grand Prix 10 Jacques Laffite 41 years , 319 days 1985 European Grand Prix Source : Podium finishes ( edit ) Total podium finishes (
edit ) Driver Seasons Entries Podiums Percentage Michael Schumacher 1991 -- 2006 , 2010 -- 2012 308 155 7001503200000000000 ♠ 50.32 % Lewis Hamilton 2007 --
2018 211 119 7001564000000000000 ♠ 56.40 % Alain Prost 1980 -- 1991 , 1993 202 106 7001524800000099999 ♠ 52.48 % Sebastian Vettel 2007 -- 2018 202 101
7001500000000000000 ♠ 50.00 % 5 Fernando Alonso 2001 , 2003 -- 2018 296 97 7001327700000000000 ♠ 32.77 % 6 Kimi Räikkönen 2001 -- 2009 ,
</item 3>

<item 4>
List of Formula One driver records
1953 Indianapolis 500 Tony Brooks 1957 British Grand Prix 9 4th race Bob Sweikert 1955 Indianapolis 500 Jacques Villeneuve 1996 European Grand Prix Source :
Most races before first win ( edit ) Start Driver Race 130th race Mark Webber 2009 German Grand Prix 123rd race Rubens Barrichello 2000 German Grand Prix 119th
race Jarno Trulli 2004 Monaco Grand Prix 113th race Jenson Button 2006 Hungarian Grand Prix 5 111th race Nico Rosberg 2012 Chinese Grand Prix 6 110th race
Giancarlo Fisichella 2003 Brazilian Grand Prix 7 96th race Mika Häkkinen 1997 European Grand Prix 8 95th race Thierry Boutsen 1989 Canadian Grand Prix 9 91st
race Jean Alesi 1995 Canadian Grand Prix 10 82nd race Eddie Irvine 1999 Australian Grand Prix Valtteri Bottas 2017 Russian Grand Prix Sources : Most races
without a win ( edit ) Driver Entries Starts Best result Andrea de Cesaris 214 208 2nd Nick
</item 4>

<item 5>
List of Formula One driver records
Piquet 1987 Monaco -- 1987 Portuguese Michael Schumacher 2000 -- 2001 2000 Hungarian -- 2001 Brazilian 2004 European -- 2004 Italian Lewis Hamilton 2007 2007
Australian -- 2007 British Nico Rosberg 2015 2015 Australian -- 2015 British 2016 2016 Belgian -- 2016 Abu Dhabi Source : Most consecutive podium finishes from
first race of season ( edit ) Driver Season Consecutive podium finishes Podiums Michael Schumacher 2002 Australian -- Japanese 17 Fernando Alonso 2006 Bahrain
-- Canadian 9 Lewis Hamilton 2007 Australian -- British 2015 Australian -- British Sebastian Vettel 2011 Australian -- British Nico Rosberg 2015 Australian --
British 7 Nico Rosberg 2014 Australian -- Austrian 8 8 Niki Lauda 1976 Brazilian -- Swedish 7 Alain Prost 1988 Brazilian -- French Michael Schumacher 1994
Brazilian -- French Jenson Button 2009 Australian -- Turkish Source needed Youngest Drivers to score a podium finish ( edit ) ( only the first podium finish for
each driver is listed ) Driver Age Place Race Max
</item 5>

<item 6>
List of Formula One driver records
StatsF1.com . Retrieved 13 November 2016 . ^ Jump up to : `` Most wins & podiums for same team '' . StatsF1.com . Jump up ^ `` Most wins different constructors
'' . StatsF1.com . Retrieved 12 May 2015 . Jump up ^ `` Least wins in World Championship winning year '' . StatsF1.com . Retrieved 27 May 2015 . Jump up ^ ``
Most grand prix without podium '' . StatsF1.com . Retrieved 27 May 2015 . Jump up ^ `` Least career wins by World Champion '' . StatsF1.com . Retrieved 27 May
2015 . Jump up ^ `` The 2016 season in numbers '' . Formula1.com. 16 December 2016 . Retrieved 14 February 2018 . Jump up ^ `` Grands Prix before podium '' .
StatsF1.com . Retrieved 21 September 2017 . Jump up ^ `` Most 2nd places '' . StatsF1.com . Retrieved 26 August 2015 .
</item 6>

<item 7>
List of Formula One driver records
( edit ) ( only the first podium finish for each driver is listed ) Driver Age Place Race Max Verstappen 18 years , 228 days 1st 2016 Spanish Grand Prix Lance
Stroll 18 years , 239 days 3rd 2017 Azerbaijan Grand Prix Sebastian Vettel 21 years , 73 days 1st 2008 Italian Grand Prix Daniil Kvyat 21 years , 91 days 2nd
2015 Hungarian Grand Prix 5 Kevin Magnussen 21 years , 162 days 2nd 2014 Australian Grand Prix 6 Fernando Alonso 21 years , 237 days 3rd 2003 Malaysian Grand
Prix 7 Robert Kubica 21 years , 278 days 3rd 2006 Italian Grand Prix 8 Ralf Schumacher 21 years , 287 days 3rd 1997 Argentine Grand Prix 9 Elio de Angelis 21
years , 307 days 2nd 1980 Brazilian Grand Prix 10 Bruce McLaren 21 years ,
</item 7>

<item 8>
List of Formula One driver records
120 days 3rd 1953 Italian Grand Prix 10 Jack Brabham 44 years , 107 days 2nd 1970 British Grand Prix Source : Most career starts without a podium ( edit )
Driver Seasons Entries Starts Best result Nico Hülkenberg 2010 , 2012 -- 2018 140 138 4th Adrian Sutil 2007 -- 2011 , 2013 -- 2014 128 128 4th Pierluigi Martini
1984 -- 1985 , 1988 -- 1995 124 119 4th Philippe Alliot 1984 -- 1990 , 1993 -- 1994 116 109 5th 5 Pedro Diniz 1995 -- 2000 99 98 5th 6 Ukyo Katayama 1992 --
1997 97 95 5th 7 Jonathan Palmer 1983 -- 1989 88 83 4th 8 Marc Surer 1979 -- 1986 82 4th 9 Vitantonio Liuzzi 2005 -- 2007 , 2009 --
</item 8>

<item 9>
List of Formula One driver records
2004 -- 2017 269 11 41 7 Riccardo Patrese 1977 -- 1993 256 6 37 8 Jacques Laffite 1974 -- 1986 176 6 32 Jean Alesi 1989 -- 2001 201 10 Juan Pablo Montoya 2001
-- 2006 94 7 30 Source : Most races before scoring a podium finish ( edit ) Entry Driver Race 91st race Martin Brundle 1992 French Grand Prix 73rd race Mika
Salo 1999 German Grand Prix 68th race Jenson Button 2004 Malaysian Grand Prix 67th race Johnny Herbert 1995 Spanish Grand Prix Pedro de la Rosa 2006 Hungarian
Grand Prix 6 60th race Gianni Morbidelli 1995 Australian Grand Prix 7 57th race Felipe Massa 2006 European Grand Prix 8 56th race Mark Webber 2005 Monaco Grand
Prix 9 55th race Kamui Kobayashi 2012 Japanese Grand Prix Daniel Ricciardo 2014 Spanish Grand Prix Source : Points ( edit ) Throughout the history of the World
Championship ,
</item 9>

<item 10>
List of Formula One driver records
positions Youngest Oldest Most races without Fastest laps Total Percentage In one season Highest percentage in one season Youngest Oldest Most at the same GP
Podium finishes Total Percentage In one season Consecutive Consecutive from first race of season Youngest Oldest Most career starts without a podium Most career
podiums without a win Most career podiums without a World Championship Most races before scoring a podium finish Points Career points Total races finished in
the points Consecutive races finished in the points Highest average points per race started Highest percentage of races finished in the points Most in one
season Youngest Oldest Most points without a win Most without being World Champion World Champions with fewest career points Most races before scoring points
Most races without scoring points Race leaders Every lap , total For at least one lap , total Youngest Most laps Longest distance Most laps without a win Most
consecutive laps in the lead Multiple driver records Pole and win in same race Most wins from pole position in a season Pole , win and fastest lap in same race
Pole , win , fastest lap ,
</item 10>
=== END ITEM id=aa0b8fd2d63d213d ===

=== ITEM id=e200558d2fbac5a1 ===
Gold sentences and facts:
Sentence 0: Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995
  Fact 0.0: Mumbai has the Marathi name मुंबई.
  Fact 0.1: Mumbai was previously named Bombay.
  Fact 0.2: Mumbai was renamed in 1995.

Retrieved items:
<item 1>
Renaming of cities in India
For example , the change of Madras ( Tamil : மதராஸ் Madras ) to Chennai ( Tamil : சென்னை Chennai ) was reflected in many of India 's languages , and
incidentally in English , while the Tamil endonym had always been Chennai and remained unaffected by the change . Renaming in English ( edit ) Change in
official English spelling ( edit ) The renaming of cities is often specifically from English to Indian English in connection with that dialect 's internal
reforms . In other words , the city itself is not actually renamed in the local language , and the local name ( or endonym ) in the indigenous languages of
India does not change , but the official spelling in Indian English is amended . An example is the change from English `` Calcutta '' to English `` Kolkata ''
-- the local Bengali name ( কলকাতা Kôlkata ) did not change . Such changes in English spelling may be in order to better reflect a more accurate phonetic
transliteration of the local name , or may be for other reasons .
</item 1>

<item 2>
Renaming of cities in India
Renaming of cities in India - Wikipedia Renaming of cities in India Jump to : navigation , search The renaming of cities in India started in 1947 following the
end of the British imperial period . Several changes were politically controversial , and not all proposed changes were implemented . Each had to be approved by
the central government in New Delhi . The renaming of states and territories in India has also taken place , but until recently with actual substantial name
changes in both local language and in English such as the old British state name of Travancore - Cochin to Kerala ( 1956 ) . The most notable recent exceptions
are Indian English spelling - changes of Orissa to Odisha ( March 2011 ) and the Union Territory of Pondicherry ( which includes the City of Pondicherry ) to
Puducherry .
</item 2>

<item 3>
Renaming of cities in India
In the early years after Indian independence , many name changes were effected in northern India for English spellings of Hindi place names that had simply been
romanised inconsistently by the British administration -- such as the British spelling `` Jubbulpore , '' renamed `` Jabalpur '' ( जबलपुर ) among the first
changes in 1947 . These changes did not generate significant controversy . More recent and high - profile changes -- including renaming such major cities as
Calcutta to Kolkata -- have generated greater controversy . Since independence , such changes have typically been enacted officially by legislation at local or
national Indian government level , and may or may not then be adopted by the Indian media , particularly the influential Indian press . In the case of smaller
towns and districts which were less notable outside and inside India , and where a well known English name ( or exonym ) could not be said to exist , older
spellings used under British India may not have had any specific legislation other than changes in practice on the romanisation of indigenous Indian language
names .
</item 3>

<item 4>
Renaming of cities in India
It is an indigenizing toponymic '' Jump up ^ Reserve Bank of India 's instructions for banks & banking operations Reserve Bank of India 2001 Page 713 `` The new
name `` Mumbai '' should be reflected in both English and Hindi and the change in name is to be brought about in all official communications , name plates ,
sign boards , office seals , rubber stamps , etc . '' Jump up ^ Perveez Mody The Intimate State : Love - Marriage and the Law in Delhi Page 59 - 2008 ``
Throughout this book , I refer to India 's commercial capital as Bombay rather than Mumbai ... I am well aware of the name - change effected by an Act of the
Indian Parliament in 1997 that made the city officially ' Mumbai ' ... It is the same convention I adopt when referring to Calcutta rather than Kolkata . ''
Jump up ^ Pingali Sailaja Indian English Page 16 2009 `` Bombay is now called Mumbai , Madras is now Chennai and Calcutta is Kolkata , in an attempt to
de-anglicise them . In this work , the earlier names are retained since these names were used during the period that we mostly cover . ''
</item 4>

<item 5>
Renaming of cities in India
Madhya Bharat → Madhya Pradesh ( 1 November 1959 ) Madras State → Tamil Nadu ( 14 January 1969 ) Mysore State → Karnataka ( 1 November 1973 ) Uttaranchal →
Uttarakhand ( 1 January 2007 ) Orissa → Odisha ( 4 November 2011 ) Cities ( edit ) Notable city names that were officially changed by legislation after
independence include : Jabalpur ( Hindi : जबलपुर ) , from Jubbulpore , respelled in 1947 Jajmau ( Hindi : जाजमऊ ) , from Jajesmow , respelled in 1948 Kanpur (
Hindi : कानपुर ) , from Cawnpore , respelled in 1948 Vadodara ( Gujarati : વડોદરા ) , from Baroda , respelled in 1974 Thiruvananthapuram ( Malayalam :
തിരുവനന്തപുരം ) , from Trivandrum ,
</item 5>

<item 6>
Renaming of cities in India
from Trivandrum , respelled in 1991 Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995 Kochi ( Malayalam : കൊച്ചി ) , from Cochin , respelled in 1996
Chennai ( Tamil : சென்னை ) , from Madras , renamed in 1996 Kolkata ( Bengali : কলকাতা ) , from Calcutta , respelled in 2001 Kadapa ( Telugu : కడప ) , from
Cuddapah , respelled in 2005 Puducherry ( Tamil : புதுச்சேரி ) , from Pondicherry , renamed in 2006 Bengaluru ( Kannada : ಬೆಂಗಳೂರು ) , from Bangalore ,
</item 6>

<item 7>
Renaming of cities in India
Realignment of the official Indian English name to an alternative local name ( edit ) Aside from changes to the official English spellings of local names there
have also been renaming proposals to realign the official name , hence the English name with an alternative local name . Ethnically sensitive examples include
the proposals by the Bharatiya Janata Party ( 1990 , 2001 ) to rename Ahmedabad to Karnavati and Allahabad to Prayag . These two proposals are changes from the
Muslim ( or Mughal ) name to a Hindu native name . These can be represented as a change from Urdu language to Hindi language , but since the two languages are
variants of Hindustani the proposal is effectively a cultural and ethno - religious proposal rather than a linguistic one . Adoption of name changes in India
and overseas ( edit ) Official name changes naturally take place quickly if not immediately in official government sources . Adoption may be slower among the
media in India and abroad , and among Indian authors . Important examples ( edit ) Main article : List of renamed places in India States ( edit ) Travancore -
Cochin → Kerala ( 1 November 1956 ) .
</item 7>

<item 8>
Renaming of cities in India
Jump up ^ Calcutta : A Cultural and Literary History - Page 3 Krishna Dutta - 2003 `` nationalist stance , like Bombay , which changed its name to Mumbai , or
Madras , which has become the unrecognisable Chennai , Calcutta has preferred a comparatively minor name change , which frankly is a bit of a multicultural
mishmash . '' Jump up ^ `` David Rumsey : Geographical Searching with MapRank Search ( beta ) '' . Rumsey.mapranksearch.com . Retrieved 2012 - 08 - 15 . Jump up
^ Beam , Christopher ( 2006 - 07 - 12 ) . `` How Bombay became Mumbai . - Slate Magazine '' . Slate.com . Retrieved 2012 - 08 - 15 .
</item 8>

<item 9>
Renaming of cities in India
The name changes are not without controversy among Indians . In several instances , the name change represents a struggle between a cosmopolitan elite and a
local , regional - language populace over defining the city in ways that go far beyond a simple change of name . '' Jump up ^ Steven I. Wilkinson Votes and
Violence : Electoral Competition and Ethnic Riots in India 2006 Page 23 `` The BJP proposed in 1990 and 2001 that Ahmedabad be renamed `` Karnavati . '' Hindu ,
June 1 1 , 2001 . Similar proposals have been made to rename Allahabad `` Prayag . '' Jump up ^ Cosmopolitanism - Page 73 Carol A. Breckenridge , Sheldon
Pollock , Homi K. Bhabha - 2002 `` In one sense , the decision to officialise the name Mumbai is part of a widespread Indian pattern of replacing names
associated with colonial rule with names associated with local , national , and regional heroes .
</item 9>

<item 10>
Renaming of cities in India
Other changes were only changes in some of the indigenous languages . For example , the renaming of the Madras Presidency to Madras State in 1947 and then Tamil
Nadu in 1969 required non-Tamil speakers to change from an approximation of the British name ( Tamil : மதராஸ் மாகாணம் Madras Presidency , then Madras State
Tamil : மதராஸ் மாநிலம் ) to a native Tamil name ( Tamil : தமிழ்நாடு Tamil Nadu , `` Tamil land '' ) . In general , changes to the local names of cities in the
indigenous languages are less common . However , a change in English may sometimes also be a reflection of changes in other Indian languages other than the
specific local one .
</item 10>
=== END ITEM id=e200558d2fbac5a1 ===

=== ITEM id=05504de89653bb22 ===
Gold sentences and facts:
Sentence 0: The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet Bandits and must outrun and out - prank them again .
  Fact 0.0: Kevin and his family decide to take a trip to Florida.
  Fact 0.1: Kevin takes the wrong plane.
  Fact 0.2: Kevin ends up in New York City.
  Fact 0.3: Kevin tries to make do with what he has.
  Fact 0.4: Kevin uses his father's credit card to stay at the Plaza Hotel.
  Fact 0.5: Kevin is soon confronted by the Wet Bandits.
  Fact 0.6: Kevin must outrun and out-prank the Wet Bandits again.

Retrieved items:
<item 1>
Home Alone 2: Lost in New York
Peter suddenly calls out , `` Kevin , you spent $967 on room service ? ! '' at which point Kevin runs off . Cast ( edit ) Main cast ( edit ) Macaulay Culkin as
Kevin McCallister Joe Pesci as Harry Lime Daniel Stern as Marv Merchants The McCallister family ( edit ) Catherine O'Hara as Kate McCallister John Heard as
Peter McCallister Devin Ratray as Buzz McCallister , Kevin 's brother who gets him into trouble Kieran Culkin ( Macaulay Culkin 's younger brother ) as Fuller
McCallister Gerry Bamman as Frank McCallister , Kevin 's uncle Hillary Wolf as Megan McCallister Maureen Elisabeth Shay ( who replaces Angela Goethals ) as
Linnie McCallister Mike Maronna as Jeff McCallister Terrie Snell as Leslie McCallister , Kevin 's aunt Jedidiah Cohen as Rod McCallister Senta Moses as Tracy
McCallister Daiana Campeanu as Sondra McCallister Anna Slotky as Brooke McCallister Other characters ( edit ) Brenda Fricker as the Pigeon Lady Tim Curry as Mr.
</item 1>

<item 2>
Home Alone 2: Lost in New York
Disc One Home Alone ( Main Title ) ( 2 : 07 ) This Year 's Wish ( 1 : 47 ) We Overslept Again / Holiday Flight ( 3 : 19 ) Separate Vacations * ( 1 : 58 )
Arrival in New York * * ( 2 : 59 ) The Thieves Return ( 3 : 28 ) Plaza Hotel ( 3 : 04 ) Concierge ( 1 : 31 ) Distant Goodnights ( Christmas Star ) ( Lyrics by
Leslie Bricusse ) ( 2 : 05 ) A Day in the City ( : 59 ) Duncan 's Toy Store ( 2 : 41 ) Turtle Doves ( 1 : 29 ) To the Plaza , Presto ( 3 : 27 ) Race to the Room
/ Hot Pursuit ( 4 : 08 ) Haunted Brownstone ( 3 : 02 ) Appearance of the Pigeon Lady ( 3 : 21 ) Christmas at Carnegie Hall ( 5 : 15 ) O Come ,
</item 2>

<item 3>
Home Alone 2: Lost in New York
Mom ( 1983 ) National Lampoon 's Vacation ( 1983 ) Nate and Hayes ( 1983 ) National Lampoon 's European Vacation ( 1985 ) Pretty in Pink ( 1986 ) Some Kind of
Wonderful ( 1987 ) The Great Outdoors ( 1988 ) National Lampoon 's Christmas Vacation ( 1989 ) Home Alone ( 1990 ) Career Opportunities ( 1991 ) Dutch ( 1991 )
Beethoven ( 1992 ) Home Alone 2 : Lost in New York ( 1992 ) Dennis the Menace ( 1993 ) Baby 's Day Out ( 1994 ) Miracle on 34th Street ( 1994 ) 101 Dalmatians (
1996 ) Flubber ( 1997 ) Home Alone 3 ( 1997 ) Reach the Rock ( 1998 ) Just Visiting ( 2001 ) Maid in Manhattan ( 2002 , story ) Drillbit Taylor ( 2008 ,
original character ) Related Do n't You Forget About Me ( 2009 documentary ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 3>

<item 4>
Home Alone 2: Lost in New York
When the duo chase Kevin around the townhouse , he escapes and calls the police . Harry and Marv catch him and take him down to Central Park to kill him , but
the pigeon lady sneaks in and throws birdseed on the duo , and the pigeons storm in to incapacitate them . Kevin sets off fireworks he had bought earlier to
signal the police . The police arrive and arrest the bandits . At the toy store , Mr. Duncan finds a note from Kevin attached to the brick , explaining about
what happened . Kate , still searching for Kevin on foot , remembers his fondness for Christmas trees . After observing Kevin making a wish at the Rockefeller
Center Christmas Tree , Kate meets him there , and they reconcile . On Christmas Day , a truckload of gifts arrive at the McCallisters ' hotel room from the toy
store . Kevin and Buzz reconcile , and Buzz allows him to open up the first present . Kevin goes to Central Park to give the pigeon lady the second turtledove .
At the Plaza , Buzz receives the bill for Kevin 's stay from Cedric and shows it to Peter .
</item 4>

<item 5>
Home Alone 2: Lost in New York
While Buzz fakes an apology to Kevin in front of the McCallisters ( who are convinced it is sincere ) , Kevin refuses to apologize to Buzz and storms to the
third floor of the house . Prior to the pageant , Peter has unknowingly caused the alarm clock to reset ; consequently , the family once again oversleeps . In
the confusion they have to rush to reach the airport on time , Kevin falls behind everyone else , and mistakenly boards a flight bound for New York City while
trying to replace the batteries for a Talkboy , carrying Peter 's bag containing his wallet and a large amount of money ; upon arrival in Miami , Kate realizes
that Kevin is missing again . In New York , Kevin tours the city and meets a homeless woman tending to pigeons at Central Park , where he runs off . Kevin goes
to the Plaza Hotel , where he tantalizes the staff into checking in with Peter 's credit card . On Christmas Eve , Kevin tours the city in a limousine and
visits a toy store where he meets its philanthropic owner , Mr. Duncan .
</item 5>

<item 6>
Home Alone 2: Lost in New York
It is the second film in the Home Alone series and the sequel to Home Alone . Macaulay Culkin reprises his role as Kevin McCallister , while Joe Pesci and
Daniel Stern reprise their roles as the Wet Bandits , now known as the Sticky Bandits . Catherine O'Hara , John Heard , Rob Schneider , Tim Curry , and Brenda
Fricker are also featured . The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New
York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet
Bandits and must outrun and out - prank them again . The film was shot in Winnetka , Illinois ; O'Hare International Airport in Chicago ; Evanston , Illinois ;
and New York City ( which was star Culkin 's hometown at the time ) . The exterior of Duncan 's Toy Chest in New York City was filmed outside of the Rookery
Building in downtown Chicago .
</item 6>

<item 7>
Home Alone 2: Lost in New York
Duncan . Kevin learns that the proceeds from the store 's Christmas sales will be donated to a children 's hospital . Kevin donates $20 to the hospital . As a
token of appreciation , Mr. Duncan offers Kevin a pair of ceramic turtledoves as a gift , instructing him to give one to another person as a sign of eternal
friendship . After running into Harry and Marv , who have recently escaped from prison and are now called `` The Sticky Bandits '' , Kevin retreats to the Plaza
Hotel . The hotel 's concierge , Mr. Hector , confronts Kevin about the credit card , which has been reported stolen . Kevin flees after evading Mr. Hector ,
but is captured by Harry and Marv . The duo discuss plans for breaking into the toy store that night . Kevin manages to escape by pinching a woman in front of
him , which results in the woman attacking the duo . Kevin 's family travels to New York after tracking the whereabouts of the stolen credit card , and Kate
searches the city for Kevin .
</item 7>

<item 8>
Home Alone 2: Lost in New York
In the beginning of the novelization , a prologue , which ends up being Marv 's nightmare in prison , he and Harry sneak away from the cops and return to Kevin
's house to seek revenge on Kevin . Kevin bolts into the garage with Marv and Harry in hot pursuit . Harry and Marv end up triggering extra traps that Kevin had
set up in the garage . Kevin watches as Marv ends up triggering a trap where a running lawnmower falls on his head ( This was a trap featured in Home Alone 3 )
. Sequels ( edit ) Main article : Home Alone 3 A second sequel , Home Alone 3 , followed in 1997 . Two additional television movie sequels , Taking Back the
House and The Holiday Heist , would follow in 2002 and 2012 , respectively . See also ( edit ) List of films featuring fictional films References ( edit ) Jump
up ^ `` HOME ALONE 2 - LOST IN NEW YORK ( PG ) '' . British Board of Film Classification . November 4 , 2017 . Retrieved November 10 , 2015 .
</item 8>

<item 9>
Home Alone 2: Lost in New York
Meanwhile , Kevin goes to his uncle Rob 's townhouse , only to find the house vacant and undergoing renovations , while Rob and his family are in Paris . At
Central Park , he encounters and befriends the pigeon lady . They go to Carnegie Hall , where an orchestra is performing . The pigeon lady explains how her life
collapsed and how she dealt with it by taking care of the pigeons in the park . Kevin gives the pigeon lady some advice and promises that he will be her friend
. Kevin , after remembering what the bandits said , returns to the townhouse , and rigs it with numerous booby traps . Kevin arrives at the toy store during
Harry and Marv 's robbery and throws a brick through the window , setting off the store 's alarm . Kevin then lures the duo to the townhouse , where he springs
the traps and Harry and Marv suffer various injuries . When the duo chase Kevin around the townhouse , he escapes and calls the police .
</item 9>

<item 10>
Home Alone 2: Lost in New York
Contents ( hide ) 1 Plot 2 Cast 2.1 Main cast 2.2 The McCallister family 2.3 Other characters 3 Release 3.1 Marketing 3.2 Box office 3.3 Critical reception 3.4
Home media 3.5 Social Media 4 Music 4.1 Original Score 4.2 Original Soundtrack Album 4.3 The Deluxe Edition 5 Novelization 6 Sequels 7 See also 8 References 9
External links Plot ( edit ) In Chicago , the McCallister family is preparing for a Christmas vacation in Miami . On the night before their departure , the
entire family gathers at Peter and Kate 's home , where their youngest son , Kevin , sees Florida as contradictory to Christmas , on his opinion of the lack of
Christmas trees in Florida . During the school Christmas pageant , Kevin 's older brother Buzz humiliates him during his solo , causing Kevin to retaliate ,
thereby ruining the performance .
</item 10>
=== END ITEM id=05504de89653bb22 ===

=== ITEM id=01ac7ae0fbf24226 ===
Gold sentences and facts:
Sentence 0: No . Cloverway inc . title Original Japanese and Viz Media titles Episode director ( s ) Writer ( s ) Art director ( s ) Animation director ( s ) Original airdate English airdate ( United States ) 90 `` Star Struck , Bad Luck '' `` Premonition of the Apocalypse : The Mysterious New Guardians Appear '' `` Chikyū hōkai no yokan ? Nazo no shin Senshi shutsugen '' ( 地球 崩壊 の 予感 ? 謎 の 新 戦士 出現 ) Junichi Sato Sukehiro Tomita Kenichi Tajiri Masahiro Andō March 19 , 1994 June 12 , 2000 After experiencing apocalyptic visions , Rei is attacked by a powerful monster called a Daimon that Sailor Moon is unable to fight after it disables her brooch , reversing her transformation . The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the Sailor Team to fend for themselves . Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92 `` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an attractive and supposedly male student of a prestigious local school , with whom they both become infatuated . After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon . Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd never said she was male . 93 `` Bad Harmony '' `` Usagi 's Idol : The Graceful Genius Michiru '' `` Usagi no akogare ! Yūbi no tensai Michiru '' ( うさぎ の 憧れ ! 優美 な 天才 みちる ) Hiroki Shibata Megumi Sugihara Kazuyuki Hashimoto Taichi Nakamura April 23 , 1994 June 15 , 2000 When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls , she seeks advice from Michiru Kaioh , who suggests that Usagi and Mamoru go to a violin recital . 94 `` Swept Off Her Feet '' `` Protect the Pure Heart : The Three - Way Battle '' `` Pyua na kokoro wo mamore ! Teki mikata mittsu tomoe ransen '' ( 純 な 心 を 守れ ! 敵 味方 三つ巴 乱戦 ) Noriyo Sasaki Sukehiro Tomita Kenichi Tajiri Ikuko Itō April 30 , 1994 June 16 , 2000 Usagi 's friend , Unazuki , is attacked by a Daimon and has her pure heart crystal stolen . But when Sailor Moon tries to take the crystal back to Unazuki , Uranus and Neptune attempt to steal it in order to claim the Talisman . 95 `` Blinded By Love 's Light '' `` Let Moon Help With Your Love Problems '' `` Koi no otasuke wa Mūn ni omakase '' ( 恋 の おたすけ は ムーン に おまかせ ) Harume Kosaka Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga May 7 , 1994 June 19 , 2000 Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other , but find themselves targeted by a Daimon . 96 `` Lita Borrows Trouble '' `` Coldhearted Uranus : Makoto in Danger '' `` Reikoku na Uranusu ? Makoto no pinchi '' ( 冷酷 な ウラヌス ? まこと の ピンチ ) Kōnosuke Uda Yoji Enokido Kazuhisa Asai Masahiro Andō May 14 , 1994 June 20 , 2000 Makoto is targeted by a Daimon , but manages to temporarily evade it . Keen to analyze her pure heart crystal , Haruka befriends Makoto in order to get close to her . 97 `` Damp Spirits '' `` The Labyrinth of Water : Ami Targeted '' `` Mizu no rabarinsu ! Nerawareta Ami '' ( 水 の ラビリンス ! ねら われ た 亜美 ) Junichi Sato Megumi Sugihara Kazuyuki Hashimoto Katsumi Tamegai May 21 , 1994 June 21 , 2000 Ami 's desire to avoid competition leads her to offending Michiru inadvertently in a swimming race , as well as Usagi accidentally hurting Ami 's feelings . Meanwhile , a Daimon selects Ami as a target . 98 `` Friendly Foes '' `` To Save Our Friends : Moon and Uranus Join Forces '' `` Tomodachi wo sukue ! Mūn Uranusu rengō '' ( 友達 を 救え ! ムーン ウラヌス 連合 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Taichi Nakamura May 28 , 1994 June 22 , 2000 After Sailor Neptune is captured and possibly killed by Kaolinite , Sailor Moon and Sailor Uranus find themselves shackled together and forced to reassess their less - than - friendly attitudes toward each other in order to save Sailor Neptune from further harm . 99 `` Mixed Emotions '' `` A Man 's Kindness : Yuichiro Heartbroken by Rei '' `` Otoko no yasashisa ! Yūichirō Rei ni shitsuren ? '' ( 男 の 優し さ ! 雄一郎 レイ に 失恋 ? ) Harume Kosaka Megumi Sugihara Yoshiyuki Shikano Shinya Hasegawa June 18 , 1994 June 23 , 2000 Worried by Rei 's intensive meditation , Yuuichirou becomes concerned that she has fallen deeply head - over-heels in love for another man . When he sees her with Haruka and mistakes her for a man , Yuuichirou is convinced that `` he '' is Rei 's new romantic suitor . 100 `` Individual Happiness '' `` I Want to Quit Being a Sailor Guardian : Minako 's Dilemma '' `` Sērā Senshi wo yametai ! ? Minako no nayami '' ( S 戦士 を 辞め たい ! ? 美奈子 の 悩み ) Hiroki Shibata Yoji Enokido Kazuhisa Asai Mari Tominaga June 25 , 1994 June 26 , 2000 Minako is despondent about her double life of being a Sailor Guardian and a normal girl . Meanwhile , an old friend of hers , who she once fell in love with , is attacked by a Daimon , and Minako , as Sailor Venus , must use her skills of volleyball to save him . 101 `` Birthday Blues , Part 1 '' `` Usagi in Tears : a Glass Slipper for My Birthday '' `` Usagi namida ! Tanjōbi ni garasu no kutsu wo '' ( うさぎ 涙 ! 誕生 日 に ガラス の 靴 を ) Yūji Endō Katsuyuki Sumisawa Kazuyuki Hashimoto Masahiro Andō July 2 , 1994 June 27 , 2000 Usagi is furious when Mamoru forgets her 15th birthday . When Mamoru buys her an expensive pair of glass slippers to make up for it , he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal . To make matters worse , Kaolinite steals Usagi 's Cosmic Heart Compact leaving her unable to transform , and Tuxedo Mask gets captured . 102 `` Birthday Blues , Part Two '' `` The Stolen Pure Heart : Usagi 's Crisis '' `` Ubawareta pyua na kokoro ! Usagi zettai zetsumei '' ( 奪 われ た 純 な 心 ! うさぎ 絶 体 絶命 ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Taichi Nakamura July 16 , 1994 June 28 , 2000 With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped , Usagi and the four other Sailor Guardians , along with Sailors Uranus and Neptune , go to retake the compact so Usagi can become Sailor Moon again , rescue Tuxedo Mask and face Kaolinite in a final showdown at Tokyo Tower . 103 `` Hello , Sailor Mini Moon '' `` The Arrival of a Tiny Pretty Guardian '' `` Yatte kita chiccha na Bishōjo Senshi '' ( やって来 た ちっちゃな 美 少女 戦士 ) Kunihiko Ikuhara Shigeru Yanagawa Yoshiyuki Shikano Kazuya Kuroda August 6 , 1994 June 29 , 2000 Eudial , and the Witches 5 are tasked to take over from Kaolinite 's position , in which they target a genius drummer recruited to play at a festival for Rei . The Sailor Guardians can not defeat the Daimon who attacks her , but Chibiusa arrives as the Guardians ' newest ally , Sailor Chibi - Moon . 104 `` Tainted Tea Party '' `` Making New Friends : Chibi Moon 's Adventure '' `` Tomodachi wo motomete ! Chibi Mūn no katsuyaku '' ( 友達 を 求め て ! ちび ムーン の 活躍 ) Junichi Sato Megumi Sugihara Kazuhisa Asai Ikuko Itō August 20 , 1994 June 30 , 2000 Chibiusa , who has returned from Crystal Tokyo for further training , attempts to adapt to her new home by making friends her age . She tries to befriend a master who does tea ceremonies , only to become the Death Busters ' latest target . 105 `` People Who Need People '' `` I Want Power : Mako Lost in Doubt '' `` Pawā ga hoshī ! Mako - chan no mayoi michi '' ( 力 が 欲しい ! まこ ちゃん の 迷い 道 ) Hiroki Shibata Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga August 27 , 1994 July 3 , 2000 Makoto is training in the mountains , trying to become much stronger after being easily defeated by a Daimon before . Her trainer is attacked , and tells her that nobody can live alone . She attacks the Daimon , fails , and realizes she needs her best friends in order to win . 106 `` Related By Destiny '' `` The Bond of Destiny : Uranus 's Distant Past '' `` Unmei no kizuna ! Uranusu no tōi hi '' ( 運命 の きずな ! ウラヌス の 遠い 日 ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai September 3 , 1994 July 4 , 2000 Haruka remembers the past when she first met Michiru , who told her that she was destined to become Sailor Uranus . Haruka initially refused that destiny , for she loved her life too much and was afraid of change . Haruka was attacked , and Sailor Neptune was injured while defending her . In her anguish , Haruka took the transformation pen and accepted her destiny as Sailor Uranus , the Sailor Guardian of the Sky . 107 `` Art Appreciation '' `` Art Is an Explosion of Love : Chibiusa 's First Love '' `` Geijutsu wa ai no bakuhatsu ! Chibiusa no hatsukoi '' ( 芸術 は 愛 の 爆発 ! ちびうさ の 初恋 ) Yūji Endō Katsuyuki Sumisawa Minoru Ōkōchi Taichi Nakamura September 10 , 1994 July 5 , 2000 Chibiusa meets a boy named Masanori at her art class and develops a crush on him . Unfortunately , he already has a crush on Michiru , who is taking a class at that school as well . When a Daimon attacks Masanori , Chibiusa must take a stand to protect him . 108 `` Everything 's Coming Up Rosey '' `` Usagi Dancing to the Waltz '' `` Usagi no dansu wa warutsu ni notte '' ( うさぎ の ダンス は ワルツ に 乗っ て ) Harume Kosaka Megumi Sugihara Kazuhisa Asai Masahiro Andō September 17 , 1994 July 6 , 2000 An English gentleman named Edwards invites Mamoru and the girls to a fancy party , where Usagi accidentally becomes drunk and the team is forced to confront a fashionable Eudial and her latest Daimon . 109 `` No Turning Back '' `` The Shocking Moment : Everyone 's Identities Revealed '' `` Shōgeki no toki ! Akasareta tagai no shōtai '' ( 衝撃 の 刻 ! 明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ? Tarisuman shutsugen '' ( ウラヌス 達 の 死 ? タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' `` Seihai no shinpi na chikara ! Mūn nidan henshin '' ( 聖杯 の 神秘 な 力 ! ムーン 二 段 変身 ) Hiroki Shibata Sukehiro Tomita Kazuyuki Hashimoto Ikuko Itō October 22 , 1994 July 11 , 2000 Sailor Pluto appears and reawakens Sailors Uranus and Neptune , and the three talismans resonate with each other ( the third and last being Sailor Pluto 's Garnet Orb ) . The Holy Grail appears , and Sailor Moon grabs it and evolves into Super Sailor Moon , who drives Eudial out . Mimete then kills Eudial by sabotaging her car with snails . 112 `` Show Stoppers '' `` Who is the True Messiah ? Chaos of Light and Darkness '' `` Shin no kyūseishu wa dare ? Hikari to kage no kaosu '' ( 真 の 救世主 は 誰 ? 光 と 影 の カオス ) Noriyo Sasaki Shigeru Yanagawa Minoru Ōkōchi Shinya Hasegawa November 5 , 1994 July 12 , 2000 The girls end up watching the filming of a movie , where Chibiusa encounters Hotaru , whom she quickly befriends . But in the end , she is revealed to be the 12 - year - old daughter of Professor Souichi Tomoe , the possessed leader of the Death Busters . Super Sailor Moon uses her new `` Rainbow Moon Heartache '' attack to defeat the Daimon of the day . 113 `` Rini 's Risky Friendship '' `` A House Filled With Evil Presence : The Beautiful Hotaru 's Secret '' `` Yōki tadayou ie ! Bishōjo Hotaru no himitsu '' ( 妖気 漂う 家 ! 美 少女 ほたる の 秘密 ) Kōnosuke Uda Megumi Sugihara Kenichi Tajiri Taichi Nakamura November 12 , 1994 July 13 , 2000 The three Outer Guardians appear to Tuxedo Mask , and tell him why they can not fight alongside Sailor Moon and the other four Sailor Guardians . Usagi and Chibiusa visit Hotaru , and see a woman who looks like Kaolinite . The Messiah of Silence appears demanding pure hearts . 114 `` Mimet 's Mess '' `` I Love Idols : Mimete 's Dilemma '' `` Aidoru daisuki ! Nayameru Mimetto '' ( アイドル 大好き ! 悩め る ミメット ) Harume Kosaka Yoji Enokido Yoshiyuki Shikano Masahiro Andō November 19 , 1994 July 14 , 2000 Minako and Mimete enter a beauty contest at the same time , with Mimete willing to risk it all to obtain victory . Both fall short of the needed votes to win the idol position , and Mimete decides to just go back to her evil ways . 115 `` The Shadow of Silence '' `` Shadow of Silence : The Pale Glimmer of a Firefly '' `` Chinmoku no kage ! ? Awaki Hotaru hi no yurameki '' ( 沈黙 の 影 ! ? あ わき 蛍火 の ゆらめき ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai November 26 , 1994 July 17 , 2000 Hotaru overexerts herself , and is hospitalized , during which her past without friends is shown , and the reasons for it . Meanwhile , Mimete targets a soap opera actor , during which the attacking Daimon runs near Hotaru , who glows and completely freezes the Daimon . 116 `` Thorny Weather '' `` Sunny Skies After a Storm : A Friendship Dedicated to Hotaru '' `` Arashi nochi hare ! Hotaru ni sasageru yūjō '' ( 嵐 のち 晴れ ! ほたる に 捧げる 友情 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Jouji Yanase December 3 , 1994 July 18 , 2000 Mamoru takes Usagi , Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi , an old classmate of his . While there , Mimete targets Yoshiki 's pure heart and causes a storm that puts the plants at risk . Sailor Moon defeats the accompanying Daimon , U-Bara . 117 `` Heightened Hazard '' `` Higher and Stronger : A Cheer from Usagi '' `` Yori takaku yori tsuyoku ! Usagi no ōen '' ( より 高く より 強く ! うさぎ の 応援 ) Kunihiko Ikuhara Shigeru Yanagawa Minoru Ōkōchi Taichi Nakamura December 10 , 1994 July 19 , 2000 Hotaru wishes to give a letter to Shun Hayase , a famous athlete . He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians ' Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December 17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed '' `` The Messiah of Silence Awakens ? Stars of Destiny '' `` Chinmoku no Meshia kakusei ? Unmei no hoshiboshi '' ( 沈黙 の メシア の 覚せい ? 運命 の 星々 ) Harume Kosaka Yoji Enokido Kazuhisa Asai Shinya Hasegawa December 24 , 1994 September 13 , 2000 The Sailor Guardians visit a planetarium , where a man is attacked by a Daimon . The symbol of Saturn appears on Hotaru 's forehead , and she attacks the Daimon . However , Sailor Uranus , Neptune , and Pluto attack Hotaru , since she is both Sailor Saturn , the Sailor Guardian of Silence , Ruin and Birth , and also possessed by the Messiah of Silence . Super Sailor Moon takes the blow to protect Hotaru . 120 `` Next In Line '' `` An Invasion from an Another Dimension : Mystery of Infinity Academy '' `` Ijigen kara no shinryaku ! Mugen Gakuen no nazo '' ( 異 次元 から の 侵略 ! 無限 学園 の 謎 ) Kōnosuke Uda Megumi Sugihara Minoru Ōkōchi Katsumi Tamegai January 7 , 1995 July 21 , 2000 Chibiusa goes to Hotaru 's house to find that Hotaru has disappeared , and is devastated about it . At the same time , the Sailor Guardians investigate Infinity Academy , where Mimete tries one final time to defeat them before Tellu literally pulls the plug on her operation . 121 `` Fiendish Ferns '' `` A Bewitching Flower That Steals Hearts : Tellu , the Third Witch '' `` Kokoro wo ubau yōka ! Daisan no majō Teruru '' ( 心 を 奪う 妖花 ! 第 三 の 魔女 テルル ) Takuya Igarashi Shigeru Yanagawa Kazuyuki Hashimoto Taichi Nakamura January 14 , 1995 July 24 , 2000 Tellu develops a new plan to steal pure hearts , using a plant bred for that purpose . Setsuna realizes it is an alien plant made to steal pure hearts , and warns the Sailor Guardians . Tuxedo Mask causes a mutated plant to destroy Tellu , leading to them tracing her back to the mysterious Infinity Academy . 122 `` The Science Of Love '' `` Believe in Love : Ami , the Kindhearted Guardian '' `` Ai wo shinjite ! Ami , kokoro yasashiki Senshi '' ( 愛 を 信じ て ! 亜美 心 優しき 戦士 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Hideyuki Motohashi January 21 , 1995 July 25 , 2000 Ami investigates the Infinity Academy , where she meets Viluy , the fourth of the Witches 5 . Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her , destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 , Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile , Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to awaken as the Messiah of Silence . 124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る 闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei ! Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei ! Unmei no hoshiboshi betsuri no toki '' ( 新しき 生命 ! 運命 の 星々 別離 の 時 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Taichi Nakamura February 18 , 1995 July 30 , 2000 Haruka and Michiru give the newly reborn infant Hotaru to a still - healing Professor Souichi Tomoe , who is finally free of the Death Busters . However , they viciously attack Sailor Moon , for she protected Hotaru 's body and Sailor Saturn , and risked the whole entire world for one mere life . The Silver Crystal throws them back , and they kneel to her as their leader and leave the city . 127 `` Tough Kindness '' `` A Guardian 's Realization : Strength Lies Within a Pure Heart '' `` Senshi no Jikaku ! Tsuyosa wa Pyua na Kokoro no Naka ni '' ( 戦士 の 自覚 ! 強 さ は 純 な 心 の 中 に ) Harume Kosaka Sukehiro Tomita February 25 , 1995 August 1 , 2000 Chibiusa returns to the future after receiving a letter from her mother , Neo-Queen Serenity . Meanwhile , a final Daimon emerges from the rubble of Infinity Academy , and Chibiusa returns just in time to help defeat it .
  Fact 0.0: The table columns are No., Cloverway Inc. title, original Japanese and Viz Media titles, episode director(s), writer(s), art director(s), animation director(s), original airdate, and English airdate in the United States.
  Fact 0.1: Episode 90 has the Cloverway Inc. title 'Star Struck, Bad Luck', the Viz Media title 'Premonition of the Apocalypse: The Mysterious New Guardians Appear', and the original Japanese title romanized as 'Chikyū hōkai no yokan? Nazo no shin Senshi shutsugen'.
  Fact 0.2: Episode 90 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Masahiro Andō. It originally aired on March 19, 1994 and aired in English in the United States on June 12, 2000.
  Fact 0.3: After experiencing apocalyptic visions, Rei is attacked by a powerful monster called a Daimon.
  Fact 0.4: Sailor Moon is unable to fight the Daimon after it disables her brooch, which reverses her transformation.
  Fact 0.5: The Daimon is defeated by a pair of mysterious new warriors.
  Fact 0.6: The girls despondently realize that a new conflict has begun.
  Fact 0.7: Episode 91 has the Cloverway Inc. title 'Crystal Clear Again', the Viz Media title 'The Rod of Love Is Born: Usagi's New Transformation', and the original Japanese title romanized as 'Ai no roddo tanjō! Usagi no shin henshin'.
  Fact 0.8: Episode 91 was directed by Yūji Endō, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Katsumi Tamegai. It originally aired on March 26, 1994 and aired in English in the United States on June 13, 2000.
  Fact 0.9: While Mamoru strives to cheer Usagi up after her failure in the previous battle, a young girl is the latest target of the Death Busters.
  Fact 0.10: Due to Usagi's inability to transform, the under-powered Sailor Team struggles to fight off the Daimon and is once again forced to rely on the mysterious two warriors.
  Fact 0.11: Once the Sailor Team determines the girl's pure heart is not a talisman, they leave the Sailor Team to fend for themselves.
  Fact 0.12: Tuxedo Mask and Usagi hold hands after the latter's refusal to escape.
  Fact 0.13: Through the power of their love, Usagi's disabled Crystal Star brooch transforms into the Cosmic Heart Compact.
  Fact 0.14: Usagi gains the Spiral Heart Moon Rod and, along with it, the ability to transform and defeat the new enemy.
  Fact 0.15: Episode 92 has the Cloverway Inc. title 'Driving Dangerously', the Viz Media title 'A Handsome Boy? Haruka Tenoh's Secret', and the original Japanese title romanized as 'Suteki na bishōnen? Ten'ō Haruka no himitsu'.
  Fact 0.16: Episode 92 was directed by Kunihiko Ikuhara, written by Katsuyuki Sumisawa, art directed by Kazuhisa Asai, and animation directed by Hisashi Kagawa. It originally aired on April 16, 1994 and aired in English in the United States on June 14, 2000.
  Fact 0.17: Avoiding their study group, Usagi and Minako meet Haruka Tenou, an attractive and supposedly male student of a prestigious local school.
  Fact 0.18: Usagi and Minako both become infatuated with Haruka.
  Fact 0.19: After Minako is defeated in a racing game, Michiru arrives to pick up Haruka.
  Fact 0.20: Michiru and Haruka depart, with Usagi and Minako following behind them soon after.
  Fact 0.21: Once Michiru and Haruka separate at a mechanic's shop, Usagi and Minako take this as their chance.
  Fact 0.22: Haruka tells Usagi and Minako about 'him'-self until a Daimon arrives to attack Kameda, the shop mechanic.
  Fact 0.23: The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down.
  Fact 0.24: Uranus and Neptune arrive to investigate Kameda's pure heart crystal.
  Fact 0.25: When the crystal turns out not to be a talisman, Uranus and Neptune retreat.
  Fact 0.26: Sailor Moon defeats the Daimon.
  Fact 0.27: Later, the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they had told them about.
  Fact 0.28: Michiru arrives with a blonde girl, whom Usagi and Minako initially fail to realize is Haruka.
  Fact 0.29: Once Usagi and Minako realize the blonde girl is Haruka, Minako falls to the ground in anguish.
  Fact 0.30: Haruka notes that she had never said she was male.
  Fact 0.31: Episode 93 has the Cloverway Inc. title 'Bad Harmony', the Viz Media title 'Usagi's Idol: The Graceful Genius Michiru', and the original Japanese title romanized as 'Usagi no akogare! Yūbi no tensai Michiru'.
  Fact 0.32: Episode 93 was directed by Hiroki Shibata, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on April 23, 1994 and aired in English in the United States on June 15, 2000.
  Fact 0.33: When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls, she seeks advice from Michiru Kaioh.
  Fact 0.34: Michiru Kaioh suggests that Usagi and Mamoru go to a violin recital.
  Fact 0.35: Episode 94 has the Cloverway Inc. title 'Swept Off Her Feet', the Viz Media title 'Protect the Pure Heart: The Three-Way Battle', and the original Japanese title romanized as 'Pyua na kokoro wo mamore! Teki mikata mittsu tomoe ransen'.
  Fact 0.36: Episode 94 was directed by Noriyo Sasaki, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on April 30, 1994 and aired in English in the United States on June 16, 2000.
  Fact 0.37: Usagi's friend, Unazuki, is attacked by a Daimon and has her pure heart crystal stolen.
  Fact 0.38: When Sailor Moon tries to take the crystal back to Unazuki, Uranus and Neptune attempt to steal it in order to claim the Talisman.
  Fact 0.39: Episode 95 has the Cloverway Inc. title 'Blinded By Love's Light', the Viz Media title 'Let Moon Help With Your Love Problems', and the original Japanese title romanized as 'Koi no otasuke wa Mūn ni omakase'.
  Fact 0.40: Episode 95 was directed by Harume Kosaka, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on May 7, 1994 and aired in English in the United States on June 19, 2000.
  Fact 0.41: Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other.
  Fact 0.42: Naru and Umino find themselves targeted by a Daimon.
  Fact 0.43: Episode 96 has the Cloverway Inc. title 'Lita Borrows Trouble', the Viz Media title 'Coldhearted Uranus: Makoto in Danger', and the original Japanese title romanized as 'Reikoku na Uranusu? Makoto no pinchi'.
  Fact 0.44: Episode 96 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on May 14, 1994 and aired in English in the United States on June 20, 2000.
  Fact 0.45: Makoto is targeted by a Daimon, but manages to temporarily evade it.
  Fact 0.46: Haruka wants to analyze Makoto's pure heart crystal, and befriends Makoto in order to get close to her.
  Fact 0.47: Episode 97 has the Cloverway Inc. title 'Damp Spirits', the Viz Media title 'The Labyrinth of Water: Ami Targeted', and the original Japanese title romanized as 'Mizu no rabarinsu! Nerawareta Ami'.
  Fact 0.48: Episode 97 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Katsumi Tamegai. It originally aired on May 21, 1994 and aired in English in the United States on June 21, 2000.
  Fact 0.49: Ami's desire to avoid competition leads her to inadvertently offend Michiru in a swimming race.
  Fact 0.50: Usagi accidentally hurts Ami's feelings.
  Fact 0.51: Meanwhile, a Daimon selects Ami as a target.
  Fact 0.52: Episode 98 has the Cloverway Inc. title 'Friendly Foes', the Viz Media title 'To Save Our Friends: Moon and Uranus Join Forces', and the original Japanese title romanized as 'Tomodachi wo sukue! Mūn Uranusu rengō'.
  Fact 0.53: Episode 98 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on May 28, 1994 and aired in English in the United States on June 22, 2000.
  Fact 0.54: After Sailor Neptune is captured and possibly killed by Kaolinite, Sailor Moon and Sailor Uranus find themselves shackled together.
  Fact 0.55: Sailor Moon and Sailor Uranus are forced to reassess their less-than-friendly attitudes toward each other in order to save Sailor Neptune from further harm.
  Fact 0.56: Episode 99 has the Cloverway Inc. title 'Mixed Emotions', the Viz Media title 'A Man's Kindness: Yuichiro Heartbroken by Rei', and the original Japanese title romanized as 'Otoko no yasashisa! Yūichirō Rei ni shitsuren?'.
  Fact 0.57: Episode 99 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Yoshiyuki Shikano, and animation directed by Shinya Hasegawa. It originally aired on June 18, 1994 and aired in English in the United States on June 23, 2000.
  Fact 0.58: Worried by Rei's intensive meditation, Yuuichirou becomes concerned that she has fallen deeply head-over-heels in love for another man.
  Fact 0.59: When Yuuichirou sees Rei with Haruka and mistakes Haruka for a man, he is convinced that 'he' is Rei's new romantic suitor.
  Fact 0.60: Episode 100 has the Cloverway Inc. title 'Individual Happiness', the Viz Media title 'I Want to Quit Being a Sailor Guardian: Minako's Dilemma', and the original Japanese title romanized as 'Sērā Senshi wo yametai!? Minako no nayami'.
  Fact 0.61: Episode 100 was directed by Hiroki Shibata, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Mari Tominaga. It originally aired on June 25, 1994 and aired in English in the United States on June 26, 2000.
  Fact 0.62: Minako is despondent about her double life of being a Sailor Guardian and a normal girl.
  Fact 0.63: Meanwhile, an old friend of Minako's, who she once fell in love with, is attacked by a Daimon.
  Fact 0.64: Minako, as Sailor Venus, must use her skills of volleyball to save the old friend.
  Fact 0.65: Episode 101 has the Cloverway Inc. title 'Birthday Blues, Part 1', the Viz Media title 'Usagi in Tears: a Glass Slipper for My Birthday', and the original Japanese title romanized as 'Usagi namida! Tanjōbi ni garasu no kutsu wo'.
  Fact 0.66: Episode 101 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Kazuyuki Hashimoto, and animation directed by Masahiro Andō. It originally aired on July 2, 1994 and aired in English in the United States on June 27, 2000.
  Fact 0.67: Usagi is furious when Mamoru forgets her 15th birthday.
  Fact 0.68: When Mamoru buys her an expensive pair of glass slippers to make up for it, he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal.
  Fact 0.69: Kaolinite steals Usagi's Cosmic Heart Compact, leaving her unable to transform.
  Fact 0.70: Tuxedo Mask gets captured.
  Fact 0.71: Episode 102 has the Cloverway Inc. title 'Birthday Blues, Part Two', the Viz Media title 'The Stolen Pure Heart: Usagi's Crisis', and the original Japanese title romanized as 'Ubawareta pyua na kokoro! Usagi zettai zetsumei'.
  Fact 0.72: Episode 102 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on July 16, 1994 and aired in English in the United States on June 28, 2000.
  Fact 0.73: With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped, Usagi and the four other Sailor Guardians, along with Sailors Uranus and Neptune, go to retake the compact.
  Fact 0.74: They go to retake the compact so Usagi can become Sailor Moon again and rescue Tuxedo Mask.
  Fact 0.75: They face Kaolinite in a final showdown at Tokyo Tower.
  Fact 0.76: Episode 103 has the Cloverway Inc. title 'Hello, Sailor Mini Moon', the Viz Media title 'The Arrival of a Tiny Pretty Guardian', and the original Japanese title romanized as 'Yatte kita chiccha na Bishōjo Senshi'.
  Fact 0.77: Episode 103 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on August 6, 1994 and aired in English in the United States on June 29, 2000.
  Fact 0.78: Eudial and the Witches 5 are tasked to take over from Kaolinite's position.
  Fact 0.79: The Witches 5 target a genius drummer recruited to play at a festival for Rei.
  Fact 0.80: The Sailor Guardians cannot defeat the Daimon who attacks the drummer.
  Fact 0.81: Chibiusa arrives as the Guardians' newest ally, Sailor Chibi-Moon.
  Fact 0.82: Episode 104 has the Cloverway Inc. title 'Tainted Tea Party', the Viz Media title 'Making New Friends: Chibi Moon's Adventure', and the original Japanese title romanized as 'Tomodachi wo motomete! Chibi Mūn no katsuyaku'.
  Fact 0.83: Episode 104 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Ikuko Itō. It originally aired on August 20, 1994 and aired in English in the United States on June 30, 2000.
  Fact 0.84: Chibiusa has returned from Crystal Tokyo for further training.
  Fact 0.85: Chibiusa attempts to adapt to her new home by making friends her age.
  Fact 0.86: Chibiusa tries to befriend a master who does tea ceremonies, only to become the Death Busters' latest target.
  Fact 0.87: Episode 105 has the Cloverway Inc. title 'People Who Need People', the Viz Media title 'I Want Power: Mako Lost in Doubt', and the original Japanese title romanized as 'Pawā ga hoshī! Mako-chan no mayoi michi'.
  Fact 0.88: Episode 105 was directed by Hiroki Shibata, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on August 27, 1994 and aired in English in the United States on July 3, 2000.
  Fact 0.89: Makoto is training in the mountains, trying to become much stronger after being easily defeated by a Daimon before.
  Fact 0.90: Makoto's trainer is attacked, and tells her that nobody can live alone.
  Fact 0.91: Makoto attacks the Daimon and fails, and realizes she needs her best friends in order to win.
  Fact 0.92: Episode 106 has the Cloverway Inc. title 'Related By Destiny', the Viz Media title 'The Bond of Destiny: Uranus's Distant Past', and the original Japanese title romanized as 'Unmei no kizuna! Uranusu no tōi hi'.
  Fact 0.93: Episode 106 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on September 3, 1994 and aired in English in the United States on July 4, 2000.
  Fact 0.94: Haruka remembers the past when she first met Michiru, who told her that she was destined to become Sailor Uranus.
  Fact 0.95: Haruka initially refused that destiny, for she loved her life too much and was afraid of change.
  Fact 0.96: Haruka was attacked, and Sailor Neptune was injured while defending her.
  Fact 0.97: In her anguish, Haruka took the transformation pen and accepted her destiny as Sailor Uranus, the Sailor Guardian of the Sky.
  Fact 0.98: Episode 107 has the Cloverway Inc. title 'Art Appreciation', the Viz Media title 'Art Is an Explosion of Love: Chibiusa's First Love', and the original Japanese title romanized as 'Geijutsu wa ai no bakuhatsu! Chibiusa no hatsukoi'.
  Fact 0.99: Episode 107 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on September 10, 1994 and aired in English in the United States on July 5, 2000.
  Fact 0.100: Chibiusa meets a boy named Masanori at her art class and develops a crush on him.
  Fact 0.101: Masanori already has a crush on Michiru, who is taking a class at that school as well.
  Fact 0.102: When a Daimon attacks Masanori, Chibiusa must take a stand to protect him.
  Fact 0.103: Episode 108 has the Cloverway Inc. title 'Everything's Coming Up Rosey', the Viz Media title 'Usagi Dancing to the Waltz', and the original Japanese title romanized as 'Usagi no dansu wa warutsu ni notte'.
  Fact 0.104: Episode 108 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on September 17, 1994 and aired in English in the United States on July 6, 2000.
  Fact 0.105: An English gentleman named Edwards invites Mamoru and the girls to a fancy party.
  Fact 0.106: At the party, Usagi accidentally becomes drunk.
  Fact 0.107: At the party, the team is forced to confront a fashionable Eudial and her latest Daimon.
  Fact 0.108: Episode 109 has the Cloverway Inc. title 'No Turning Back', the Viz Media title 'The Shocking Moment: Everyone's Identities Revealed', and the original Japanese title romanized as 'Shōgeki no toki! Akasareta tagai no shōtai'.
  Fact 0.109: Episode 109 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on September 24, 1994 and aired in English in the United States on July 7, 2000.
  Fact 0.110: Minako becomes jealous that a Daimon has not yet attacked her, and she fears that she does not have a pure heart.
  Fact 0.111: After doing as many 'pure' things as she can, Minako is attacked, and her pure heart is taken from her.
  Fact 0.112: When Eudial nearly gets away, Usagi is forced to reveal her identity to Haruka and Michiru.
  Fact 0.113: Haruka and Michiru reveal their identities as Sailors Uranus and Neptune to Sailor Moon.
  Fact 0.114: Episode 110 has the Cloverway Inc. title 'Destiny's Arrival', the Viz Media title 'The Death of Uranus and Neptune: The Talisman Appear', and the original Japanese title romanized as 'Uranusu-tachi no shi? Tarisuman shutsugen'.
  Fact 0.115: Episode 110 was directed by Kunihiko Ikuhara, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Mari Tominaga. It originally aired on October 15, 1994 and aired in English in the United States on July 10, 2000.
  Fact 0.116: Faced with losing her position at Witches 5, Eudial creates a computer program to find the true owners of the three Talismans.
  Fact 0.117: Eudial learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts.
  Fact 0.118: Eudial and her companions take away Usagi's brooch, and follow Eudial onto a lonely island.
  Fact 0.119: The invitation turns out to be a trick, and Michiru's pure heart is taken, which turns into the mystical Deep Aqua Mirror.
  Fact 0.120: Haruka, in order to release her own talisman, shoots herself with Eudial's gun.
  Fact 0.121: Haruka's pure heart transforms into the mystical Space Sword.
  Fact 0.122: Episode 111 has the Cloverway Inc. title 'The Purity Chalice', the Viz Media title 'The Holy Grail's Mystical Power: Moon's Double Transformation', and the original Japanese title romanized as 'Seihai no shinpi na chikara! Mūn nidan henshin'.
  Fact 0.123: Episode 111 was directed by Hiroki Shibata, written by Sukehiro Tomita, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on October 22, 1994 and aired in English in the United States on July 11, 2000.
  Fact 0.124: Sailor Pluto appears and reawakens Sailors Uranus and Neptune.
  Fact 0.125: The three talismans resonate with each other, the third and last being Sailor Pluto's Garnet Orb.
  Fact 0.126: The Holy Grail appears, and Sailor Moon grabs it and evolves into Super Sailor Moon.
  Fact 0.127: Super Sailor Moon drives Eudial out.
  Fact 0.128: Mimete kills Eudial by sabotaging her car with snails.
  Fact 0.129: Episode 112 has the Cloverway Inc. title 'Show Stoppers', the Viz Media title 'Who is the True Messiah? Chaos of Light and Darkness', and the original Japanese title romanized as 'Shin no kyūseishu wa dare? Hikari to kage no kaosu'.
  Fact 0.130: Episode 112 was directed by Noriyo Sasaki, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Shinya Hasegawa. It originally aired on November 5, 1994 and aired in English in the United States on July 12, 2000.
  Fact 0.131: The girls end up watching the filming of a movie, where Chibiusa encounters Hotaru, whom she quickly befriends.
  Fact 0.132: In the end, Chibiusa is revealed to be the 12-year-old daughter of Professor Souichi Tomoe, the possessed leader of the Death Busters.
  Fact 0.133: Super Sailor Moon uses her new 'Rainbow Moon Heartache' attack to defeat the Daimon of the day.
  Fact 0.134: Episode 113 has the Cloverway Inc. title 'Rini's Risky Friendship', the Viz Media title 'A House Filled With Evil Presence: The Beautiful Hotaru's Secret', and the original Japanese title romanized as 'Yōki tadayou ie! Bishōjo Hotaru no himitsu'.
  Fact 0.135: Episode 113 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on November 12, 1994 and aired in English in the United States on July 13, 2000.
  Fact 0.136: The three Outer Guardians appear to Tuxedo Mask, and tell him why they cannot fight alongside Sailor Moon and the other four Sailor Guardians.
  Fact 0.137: Usagi and Chibiusa visit Hotaru, and see a woman who looks like Kaolinite.
  Fact 0.138: The Messiah of Silence appears demanding pure hearts.
  Fact 0.139: Episode 114 has the Cloverway Inc. title 'Mimet's Mess', the Viz Media title 'I Love Idols: Mimete's Dilemma', and the original Japanese title romanized as 'Aidoru daisuki! Nayameru Mimetto'.
  Fact 0.140: Episode 114 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Yoshiyuki Shikano, and animation directed by Masahiro Andō. It originally aired on November 19, 1994 and aired in English in the United States on July 14, 2000.
  Fact 0.141: Minako and Mimete enter a beauty contest at the same time, with Mimete willing to risk it all to obtain victory.
  Fact 0.142: Both Minako and Mimete fall short of the needed votes to win the idol position.
  Fact 0.143: Mimete decides to go back to her evil ways.
  Fact 0.144: Episode 115 has the Cloverway Inc. title 'The Shadow of Silence', the Viz Media title 'Shadow of Silence: The Pale Glimmer of a Firefly', and the original Japanese title romanized as 'Chinmoku no kage!? Awaki Hotaru hi no yurameki'.
  Fact 0.145: Episode 115 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on November 26, 1994 and aired in English in the United States on July 17, 2000.
  Fact 0.146: Hotaru overexerts herself and is hospitalized, during which her past without friends is shown, and the reasons for it.
  Fact 0.147: Meanwhile, Mimete targets a soap opera actor.
  Fact 0.148: During the attack on the actor, the attacking Daimon runs near Hotaru, who glows and completely freezes the Daimon.
  Fact 0.149: Episode 116 has the Cloverway Inc. title 'Thorny Weather', the Viz Media title 'Sunny Skies After a Storm: A Friendship Dedicated to Hotaru', and the original Japanese title romanized as 'Arashi nochi hare! Hotaru ni sasageru yūjō'.
  Fact 0.150: Episode 116 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Jouji Yanase. It originally aired on December 3, 1994 and aired in English in the United States on July 18, 2000.
  Fact 0.151: Mamoru takes Usagi, Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi, an old classmate of Mamoru's.
  Fact 0.152: While there, Mimete targets Yoshiki's pure heart and causes a storm that puts the plants at risk.
  Fact 0.153: Sailor Moon defeats the accompanying Daimon, U-Bara.
  Fact 0.154: Episode 117 has the Cloverway Inc. title 'Heightened Hazard', the Viz Media title 'Higher and Stronger: A Cheer from Usagi', and the original Japanese title romanized as 'Yori takaku yori tsuyoku! Usagi no ōen'.
  Fact 0.155: Episode 117 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on December 10, 1994 and aired in English in the United States on July 19, 2000.
  Fact 0.156: Hotaru wishes to give a letter to Shun Hayase, a famous athlete.
  Fact 0.157: Shun Hayase becomes targeted when Hotaru goes to give the letter to him.
  Fact 0.158: Hotaru possessively blasts the enemy with a hugely strong amount of energy.
  Fact 0.159: Hotaru talks to Shun about being frail and weak, and he thanks her for talking to him.
  Fact 0.160: Episode 118 has the Cloverway Inc. title 'It's In The Cards', the Viz Media title 'The Battle Inside the Demonic Space: The Sailor Guardians' Gamble', and the original Japanese title romanized as 'Makū no tatakai! Sērā Senshi no kake'.
  Fact 0.161: Episode 118 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Masahide Yanagisawa. It originally aired on December 17, 1994 and aired in English in the United States on July 20, 2000.
  Fact 0.162: Mimete attempts to create a Daimon by herself, throwing the entire Tomoe house into another dimension.
  Fact 0.163: The Guardians enter the dimension and are defeated by a game-master Daimon named U-Ikasaman.
  Fact 0.164: Hotaru manages to assist in U-Ikasaman's defeat by the use of a bluff.
  Fact 0.165: Episode 119 has the Cloverway Inc. title 'Goodness Eclipsed', the Viz Media title 'The Messiah of Silence Awakens? Stars of Destiny', and the original Japanese title romanized as 'Chinmoku no Meshia kakusei? Unmei no hoshiboshi'.
  Fact 0.166: Episode 119 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Shinya Hasegawa. It originally aired on December 24, 1994 and aired in English in the United States on September 13, 2000.
  Fact 0.167: The Sailor Guardians visit a planetarium, where a man is attacked by a Daimon.
  Fact 0.168: The symbol of Saturn appears on Hotaru's forehead, and she attacks the Daimon.
  Fact 0.169: Sailor Uranus, Neptune, and Pluto attack Hotaru, since she is both Sailor Saturn, the Sailor Guardian of Silence, Ruin and Birth, and also possessed by the Messiah of Silence.
  Fact 0.170: Super Sailor Moon takes the blow to protect Hotaru.
  Fact 0.171: Episode 120 has the Cloverway Inc. title 'Next In Line', the Viz Media title 'An Invasion from an Another Dimension: Mystery of Infinity Academy', and the original Japanese title romanized as 'Ijigen kara no shinryaku! Mugen Gakuen no nazo'.
  Fact 0.172: Episode 120 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Katsumi Tamegai. It originally aired on January 7, 1995 and aired in English in the United States on July 21, 2000.
  Fact 0.173: Chibiusa goes to Hotaru's house to find that Hotaru has disappeared, and is devastated about it.
  Fact 0.174: At the same time, the Sailor Guardians investigate Infinity Academy.
  Fact 0.175: Mimete tries one final time to defeat the Sailor Guardians before Tellu literally pulls the plug on her operation.
  Fact 0.176: Episode 121 has the Cloverway Inc. title 'Fiendish Ferns', the Viz Media title 'A Bewitching Flower That Steals Hearts: Tellu, the Third Witch', and the original Japanese title romanized as 'Kokoro wo ubau yōka! Daisan no majō Teruru'.
  Fact 0.177: Episode 121 was directed by Takuya Igarashi, written by Shigeru Yanagawa, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on January 14, 1995 and aired in English in the United States on July 24, 2000.
  Fact 0.178: Tellu develops a new plan to steal pure hearts, using a plant bred for that purpose.
  Fact 0.179: Setsuna realizes the plant is an alien plant made to steal pure hearts, and warns the Sailor Guardians.
  Fact 0.180: Tuxedo Mask causes a mutated plant to destroy Tellu.
  Fact 0.181: This leads the Sailor Guardians to trace Tellu back to the mysterious Infinity Academy.
  Fact 0.182: Episode 122 has the Cloverway Inc. title 'The Science Of Love', the Viz Media title 'Believe in Love: Ami, the Kindhearted Guardian', and the original Japanese title romanized as 'Ai wo shinjite! Ami, kokoro yasashiki Senshi'.
  Fact 0.183: Episode 122 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Hideyuki Motohashi. It originally aired on January 21, 1995 and aired in English in the United States on July 25, 2000.
  Fact 0.184: Ami investigates the Infinity Academy, where she meets Viluy, the fourth of the Witches 5.
  Fact 0.185: Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers.
  Fact 0.186: After Viluy's battle, her own microscopic robots turn against her, destroying her in the process.
  Fact 0.187: Episode 123 has the Cloverway Inc. title 'Wake Up Call', the Viz Media title 'Shadows of Destruction: the Messiah of Silence Awakens', and the original Japanese title romanized as 'Hametsu no kage! Chinmoku no Meshia no mezame'.
  Fact 0.188: Episode 123 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Mari Tominaga. It originally aired on January 28, 1995 and aired in English in the United States on July 26, 2000.
  Fact 0.189: Cyprine is the last of the Witches 5.
  Fact 0.190: Cyprine steals pure hearts from Infinity Academy students.
  Fact 0.191: The Sailor Guardians engage Cyprine, and her alter-ego, Ptilol, in a showdown.
  Fact 0.192: Chibiusa is abducted by the recently resurrected Kaolinite.
  Fact 0.193: Chibiusa's pure heart is stolen by Mistress 9, who has taken over Hotaru's body, which allows her to awaken as the Messiah of Silence.
  Fact 0.194: Episode 124 has the Cloverway Inc. title 'Who's Really Who?', the Viz Media title 'The Coming Terror of Darkness: Struggle of the Eight Guardians', and the original Japanese title romanized as 'Semari kuru yami no kyōfu! Kusen no Hassenshi'.
  Fact 0.195: Episode 124 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on February 4, 1995 and aired in English in the United States on July 28, 2000.
  Fact 0.196: Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy.
  Fact 0.197: Sailor Moon is abducted, as she is the holder of the Holy Grail.
  Fact 0.198: After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time, Sailor Uranus and Sailor Neptune destroy Germatoid.
  Fact 0.199: Germatoid is the Daimon possessing Professor Souichi Tomoe.
  Fact 0.200: Sailor Uranus and Sailor Neptune destroy Germatoid with their talismans, the Deep Aqua Mirror and the Space Sword.
  Fact 0.201: Episode 125 has the Cloverway Inc. title 'Darkness, My Old Friend', the Viz Media title 'The Shining Shooting Star: Saturn and the Messiah', and the original Japanese title romanized as 'Kagayaku ryūsei! Satān soshite kyūseishu'.
  Fact 0.202: Episode 125 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on February 11, 1995 and aired in English in the United States on July 29, 2000.
  Fact 0.203: Sailor Moon is tricked into giving Mistress 9 the Holy Grail, which she uses to summon Pharaoh 90 to Earth.
  Fact 0.204: Hotaru regains consciousness with the assistance of Sailor Moon and her father, destroying Mistress 9.
  Fact 0.205: Hotaru fully awakens the spirit of Sailor Saturn within herself at last.
  Fact 0.206: Sailor Saturn returns Chibiusa's pure heart crystal.
  Fact 0.207: Sailor Saturn appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers.
  Fact 0.208: Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more.
  Fact 0.209: Super Sailor Moon saves both Hotaru and Sailor Saturn.
  Fact 0.210: Sailor Saturn is reborn as a baby immediately afterward.
  Fact 0.211: Episode 126 has the Cloverway Inc. title 'Second Chance', the Viz Media title 'A New Life: Parting of the Stars of Destiny', and the original Japanese title romanized as 'Atarashiki seimei! Unmei no hoshiboshi betsuri no toki'.
  Fact 0.212: Episode 126 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Taichi Nakamura. It originally aired on February 18, 1995 and aired in English in the United States on July 30, 2000.
  Fact 0.213: Haruka and Michiru give the newly reborn infant Hotaru to a still-healing Professor Souichi Tomoe, who is finally free of the Death Busters.
  Fact 0.214: Haruka and Michiru viciously attack Sailor Moon, for she protected Hotaru's body and Sailor Saturn, and risked the whole entire world for one mere life.
  Fact 0.215: The Silver Crystal throws Haruka and Michiru back.
  Fact 0.216: Haruka and Michiru kneel to Sailor Moon as their leader and leave the city.
  Fact 0.217: Episode 127 has the Cloverway Inc. title 'Tough Kindness', the Viz Media title 'A Guardian's Realization: Strength Lies Within a Pure Heart', and the original Japanese title romanized as 'Senshi no Jikaku! Tsuyosa wa Pyua na Kokoro no Naka ni'.
  Fact 0.218: Episode 127 was directed by Harume Kosaka and written by Sukehiro Tomita. It originally aired on February 25, 1995 and aired in English in the United States on August 1, 2000.
  Fact 0.219: Chibiusa returns to the future after receiving a letter from her mother, Neo-Queen Serenity.
  Fact 0.220: Meanwhile, a final Daimon emerges from the rubble of Infinity Academy.
  Fact 0.221: Chibiusa returns just in time to help defeat the final Daimon.

Retrieved items:
<item 1>
List of Sailor Moon S episodes
Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked
into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her
father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart
crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor
Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby
immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei !
</item 1>

<item 2>
List of Sailor Moon S episodes
Mūn nidan henshin '' ( 聖杯 の 神秘 な 力 ! ムーン 二 段 変身 ) Hiroki Shibata Sukehiro Tomita Kazuyuki Hashimoto Ikuko Itō October 22 , 1994 July 11 , 2000 Sailor Pluto
appears and reawakens Sailors Uranus and Neptune , and the three talismans resonate with each other ( the third and last being Sailor Pluto 's Garnet Orb ) .
The Holy Grail appears , and Sailor Moon grabs it and evolves into Super Sailor Moon , who drives Eudial out . Mimete then kills Eudial by sabotaging her car
with snails . 112 `` Show Stoppers '' `` Who is the True Messiah ? Chaos of Light and Darkness '' `` Shin no kyūseishu wa dare ? Hikari to kage no kaosu '' ( 真
の 救世主 は 誰 ?
</item 2>

<item 3>
List of Sailor Moon S episodes
After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once
Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon
arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when
Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon .
Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a
blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd
never said she was male .
</item 3>

<item 4>
List of Sailor Moon S episodes
Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch
transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92
`` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko
Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an
attractive and supposedly male student of a prestigious local school , with whom they both become infatuated .
</item 4>

<item 5>
List of Sailor Moon S episodes
タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial
creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within
their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's
pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her
pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' ``
Seihai no shinpi na chikara !
</item 5>

<item 6>
List of Sailor Moon S episodes
Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her ,
destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no
mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 ,
Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile ,
Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to
awaken as the Messiah of Silence . 124 `` Who 's Really Who ? ''
</item 6>

<item 7>
List of Sailor Moon S episodes
He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about
being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians '
Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December
17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the
dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed
'' `` The Messiah of Silence Awakens ?
</item 7>

<item 8>
List of Sailor Moon S episodes
明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not
yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart
is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors
Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ?
Tarisuman shutsugen '' ( ウラヌス 達 の 死 ?
</item 8>

<item 9>
List of Sailor Moon S episodes
124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る
闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to
create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing
Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the
Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining
Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei !
</item 9>

<item 10>
List of Sailor Moon S episodes
The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear
Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō
Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous
battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight
off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the
Sailor Team to fend for themselves .
</item 10>
=== END ITEM id=01ac7ae0fbf24226 ===

=== ITEM id=8a5697765ddd4efe ===
Gold sentences and facts:
Sentence 0: The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly .
  Fact 0.0: The Roosevelt Corollary was an addition to the Monroe Doctrine.
  Fact 0.1: President Theodore Roosevelt articulated the Roosevelt Corollary.
  Fact 0.2: Theodore Roosevelt articulated the Roosevelt Corollary in his State of the Union address in 1904.
  Fact 0.3: The Roosevelt Corollary was articulated after the Venezuela Crisis of 1902-03.
  Fact 0.4: The Roosevelt Corollary states that the United States will intervene in conflicts between European countries and Latin American countries.
  Fact 0.5: Under the Roosevelt Corollary, the United States intervenes to enforce legitimate claims of the European powers.
  Fact 0.6: Under the Roosevelt Corollary, the United States intervenes rather than having the Europeans press their claims directly.

Retrieved items:
<item 1>
Truman Doctrine
Long - term policy and metaphor ( edit ) The Truman Doctrine underpinned American Cold War policy in Europe and around the world . In the words of historian
James T. Patterson , `` The Truman Doctrine was a highly publicized commitment of a sort the administration had not previously undertaken . Its sweeping
rhetoric , promising that the United States should aid all ' free people ' being subjugated , set the stage for innumerable later ventures that led to
globalistic commitments . It was in these ways a major step . '' The doctrine endured , historian Dennis Merill argues , because it addressed a broader cultural
insecurity regarding modern life in a globalized world . It dealt with Washington 's concern over communism 's domino effect , it enabled a media - sensitive
presentation of the doctrine that won bipartisan support , and it mobilized American economic power to modernize and stabilize unstable regions without direct
military intervention . It brought nation - building activities and modernization programs to the forefront of foreign policy . The Truman Doctrine became a
metaphor for emergency aid to keep a nation from communist influence .
</item 1>

<item 2>
Roosevelt Corollary
The Dominican experiment , like most other `` dollar diplomacy '' arrangements , proved temporary and untenable , and the United States launched a larger
military intervention in 1912 . U.S. Presidents also cited the Roosevelt Corollary as justification for U.S. intervention in Cuba ( 1906 -- 1909 ) , Nicaragua (
1909 -- 1910 , 1912 -- 1925 and 1926 -- 1933 ) , Haiti ( 1915 -- 1934 ) , and the Dominican Republic ( 1916 -- 1924 ) . Shift to the `` Good Neighbor '' policy
( edit ) Main article : Good Neighbor policy In 1928 , under President Calvin Coolidge , the Clark Memorandum -- often seen as a partial repudiation of the
Roosevelt Corollary -- stated that the U.S. did not have the right to intervene when there was a threat by European powers . Herbert Hoover also helped move the
U.S. away from the imperialist tendencies of the Roosevelt Corollary by going on good - will tours , withdrawing troops from Nicaragua and Haiti , and
abstaining from intervening in the internal affairs of neighboring countries .
</item 2>

<item 3>
Roosevelt Corollary
Serge Ricard of the University of Paris goes even further , stating that the Roosevelt Corollary was not merely an addendum to the earlier Monroe Doctrine ,
through which the U.S. pledged to protect the Americas from European imperialist interventions . Rather , the Roosevelt Corollary was `` an entirely new
diplomatic tenet that epitomized his ' big stick ' approach to foreign policy . '' In other words , while the Monroe Doctrine sought to bar entry to the
European empires , the Roosevelt Corollary announced the United States intention to take their place . See also ( edit ) International relations portal
Government of the United States portal United States portal Wikisource has original text related to this article : Roosevelt Corollary History of the United
States ( 1865 -- 1918 ) United States territorial acquisitions New Imperialism Dollar Diplomacy Big Stick ideology Drago Doctrine Bush Doctrine References (
edit ) Jump up ^ Theodore Roosevelt , 26th President . Prod . David Grubin . By David Grubin and Geoffrey C. Ward . Perf . Walter LaFeber . David Grubin
Productions , Inc. , 1996 .
</item 3>

<item 4>
Roosevelt Corollary
of Nations European Union Latin America United Nations Third Border Initiative International organizations ( hide ) Doctrines , policies , concepts Presidential
doctrines Proclamation of Neutrality Monroe Roosevelt Corollary Good Neighbor policy Truman Eisenhower Kennedy Johnson Nixon Carter Reagan Clinton Bush Obama
Other doctrines Lodge Corollary Stimson Kirkpatrick Weinberger Powell Rumsfeld Wolfowitz Policies and concepts Blowback Containment Domino theory
Non-interventionism Progressive realism Rollback Special Relationship Taiwan Relations Act United States intervention in Latin America Policy Monroe Doctrine (
1823 ) Platt Amendment ( 1901 -- 04 ) Roosevelt Corollary / Big Stick ideology ( 1904 ) Good Neighbor policy ( 1933 ) Dollar diplomacy Banana Wars Wars Mexican
-- American War ( 1846 - 48 ) Spanish -- American War ( 1898 ) Border War ( 1910 -- 19 ) United States involvement in the Mexican Revolution ( 1916 -- 19 )
Overt actions and occupations Paraguay expedition ( 1858 ) First Occupation of Cuba ( 1899 -- 1902 ) Panama -- Colombia separation and Hay -- Bunau - Varilla
Treaty ( 1903 ) Second Occupation of Cuba ( 1906 -- 09 ) Occupations of Honduras Occupation of Nicaragua (
</item 4>

<item 5>
Roosevelt Corollary
This was used to justify Operation PBSUCCESS that deposed the democratically elected President Jacobo Árbenz and installed the military regime of Carlos
Castillo Armas , the first in a series of military dictators in the country . Criticism ( edit ) The argument made by Mitchener and Weidenmier in 2006 in
support of the Roosevelt Corollary to the Monroe Doctrine has been criticized on the grounds that it `` represent ( s ) the one - sided approach that some
scholars bring to the study of imperialistic and hegemonic interventions and also highlight how arguments for the general utility of imperialism are
increasingly made and accepted . '' Christopher Coyne and Stephen Davies argue that a foreign policy modeled on the Roosevelt Corollary leads to negative
consequences both in national security terms and in terms of its effect on domestic politics . Critics , such as Noam Chomsky , have argued that the Roosevelt
Corollary was merely a more explicit imperialist threat , building on the Monroe Doctrine , indicating that the U.S. would intervene not only in defense of
South American states in the face of European imperialism , but would also use its muscle to obtain concessions and privileges for American corporations .
</item 5>

<item 6>
Roosevelt Corollary
The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the
Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries
to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly . Roosevelt tied his policy to the Monroe
Doctrine , and it was also consistent with his foreign policy included in his Big Stick Diplomacy . Roosevelt stated that in keeping with the Monroe Doctrine ,
the United States was justified in exercising `` international police power '' to put an end to chronic unrest or wrongdoing in the Western Hemisphere . While
the Monroe Doctrine had sought to prevent European intervention , the Roosevelt Corollary was used to justify US intervention throughout the hemisphere . In
1934 , President Franklin D. Roosevelt renounced interventionism and established his Good Neighbor policy for the Western Hemisphere .
</item 6>

<item 7>
Roosevelt Corollary
Contents ( hide ) 1 Use 1.1 Shift to the `` Good Neighbor '' policy 2 Criticism 3 See also 4 References 5 Bibliography Use ( edit ) See also : Overseas
interventions of the United States The Roosevelt Corollary was an addition to the Monroe Doctrine ; however , it could be seen as a departure . While the Monroe
Doctrine said European countries should stay out of Latin America , the Roosevelt Corollary took this further to say he had the right to exercise military force
in Latin American countries to keep European countries out . Historian Walter LaFeber wrote ( Roosevelt ) essentially turns the Monroe Doctrine on its head and
toes says the Europeans should stay out , but the United States has the right , under the doctrine , to go in to exercise police power to keep the Europeans out
of the way .
</item 7>

<item 8>
Roosevelt Corollary
It is a very nice twist on the Monroe Doctrine , and of course , it becomes very , very important because over the next 15 to 20 years , the United States will
move into Latin America about a dozen times with military force , to the point where the United States Marines become known in the area as `` State Department
Troops '' because they are always moving in to protect State Department interests and State Department policy in the Caribbean . So what Roosevelt does here ,
by redefining the Monroe Doctrine , turns out to be very historical , and it leads the United States into a period of confrontation with peoples in the
Caribbean and Central America , that was an imperative part of American imperialism . Roosevelt first used the Corollary to act in the Dominican Republic in
1904 , which at the time was severely indebted and becoming a failed state . The United States dispatched two warships and demanded the customs house be turned
over to U.S. negotiators , who then used a percentage of the proceeds to pay foreign creditors . This model -- in which United States advisors worked to
stabilize Latin American nations through temporary protectorates , staving off European action -- became known as `` dollar diplomacy '' .
</item 8>

<item 9>
Roosevelt Corollary
In 1934 , Franklin D. Roosevelt further renounced interventionism and established his `` Good Neighbor policy '' that led to the annulment of the Platt
Amendment by the Treaty of Relations with Cuba in 1934 , and the negotiation of compensation for Mexico 's nationalization of foreign - owned oil assets in 1938
. Indeed , leaving unchallenged the emergence of dictatorships like that of Fulgencio Batista in Cuba , Rafael Leonidas Trujillo in the Dominican Republic ,
Anastasio Somoza in Nicaragua , and François Duvalier in Haiti were each considered to be `` Frankenstein dictators '' due to the mishandlings of the American
occupations in the countries . The era of the Good Neighbor Policy ended with the ramp - up of the Cold War in 1945 , as the United States felt there was a
greater need to protect the western hemisphere from Soviet influence . In 1954 , Secretary of State John Foster Dulles invoked the Monroe Doctrine and the
Roosevelt Corollary at the Tenth Pan-American Conference in Caracas , denouncing the intervention of Soviet Communism in Guatemala .
</item 9>

<item 10>
Truman Doctrine
George F. Kennan ( 1904 - 2005 ) proposed the doctrine of containment in 1946 . In February 1946 , Kennan , an American diplomat in Moscow , sent his famed ``
Long Telegram '' , which predicted the Soviets would only respond to force and that the best way to handle them would be through a long - term strategy of
containment , that is stopping their geographical expansion . After the British warned that they could no longer help Greece , and following Prime Minister
Konstantinos Tsaldaris 's visit to Washington in December 1946 to ask for American assistance , the U.S. State Department formulated a plan . Aid would be given
to both Greece and Turkey , to help cool the long - standing rivalry between them . American policy makers recognized the instability of the region , fearing
that if Greece was lost to communism , Turkey would not last long . Similarly , if Turkey yielded to Soviet demands , the position of Greece would be endangered
. A regional domino effect threat therefore guided the American decision .
</item 10>
=== END ITEM id=8a5697765ddd4efe ===
