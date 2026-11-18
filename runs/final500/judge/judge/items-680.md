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

This file has 7 items.

=== ITEM id=bf3652313595edd3 ===
Gold sentences and facts:
Sentence 0: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 )
  Fact 0.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 0.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 0.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 0.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 0.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 0.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 0.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 0.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 0.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 0.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 0.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 0.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 0.12: Bart Cummings won the Melbourne Cup in 2008.
Sentence 1: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman ( 1989 , 1992 , 1995 , 2004 , 2005 )
  Fact 1.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 1.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 1.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 1.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 1.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 1.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 1.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 1.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 1.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 1.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 1.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 1.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 1.12: Bart Cummings won the Melbourne Cup in 2008.
  Fact 1.13: Etienne de Mestre has 5 Melbourne Cup wins.
  Fact 1.14: Etienne de Mestre won the Melbourne Cup in 1861.
  Fact 1.15: Etienne de Mestre won the Melbourne Cup in 1862.
  Fact 1.16: Etienne de Mestre won the Melbourne Cup in 1867.
  Fact 1.17: Etienne de Mestre won the Melbourne Cup in 1877.
  Fact 1.18: Etienne de Mestre won the Melbourne Cup in 1878.
  Fact 1.19: Lee Freedman has 5 Melbourne Cup wins.
  Fact 1.20: Lee Freedman won the Melbourne Cup in 1989.
  Fact 1.21: Lee Freedman won the Melbourne Cup in 1992.
  Fact 1.22: Lee Freedman won the Melbourne Cup in 1995.
  Fact 1.23: Lee Freedman won the Melbourne Cup in 2004.
  Fact 1.24: Lee Freedman won the Melbourne Cup in 2005.
Sentence 2: 1877 is also the year that the trainer Etienne de Mestre won his fourth Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell ( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who , though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth Melbourne Cup in 1975 . Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 .
  Fact 2.0: In 1877, the trainer Etienne de Mestre won his fourth Melbourne Cup.
  Fact 2.1: In 1877, Etienne de Mestre won with Chester.
  Fact 2.2: Chester was owned by Hon. James White.
  Fact 2.3: In 1878, as in previous years, Etienne de Mestre fielded more than one horse.
  Fact 2.4: Etienne de Mestre entered the favourite Firebell in 1878.
  Fact 2.5: Firebell was owned by W.S. Cox.
  Fact 2.6: Firebell finished last in 1878.
  Fact 2.7: Etienne de Mestre entered Chester in 1878.
  Fact 2.8: Chester was owned by Hon. James White.
  Fact 2.9: Chester was the previous year's winner.
  Fact 2.10: Chester fell in 1878.
  Fact 2.11: Etienne de Mestre entered Calamia in 1878.
  Fact 2.12: Calamia was owned by Etienne de Mestre.
  Fact 2.13: Calamia was less fancied.
  Fact 2.14: Calamia won easily by two lengths in 1878.
  Fact 2.15: First prize in 1878 was £1,790.
  Fact 2.16: The crowd in 1878 was 80,000.
  Fact 2.17: There were 30 starters in 1878.
  Fact 2.18: De Mestre's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won.
  Fact 2.19: De Mestre's record of 5 Melbourne Cups was not matched for nearly 100 years.
  Fact 2.20: Bart Cummings won his fifth Melbourne Cup in 1975.
  Fact 2.21: Bart Cummings is regarded as the best Australian horse trainer of all time.
  Fact 2.22: Bart Cummings went on to win 12 Melbourne Cups, up to 2008.

Retrieved items:
<item 1>
Melbourne Cup
The first jockey of Indigenous heritage to ride a Melbourne Cup winner was Frank Reys in 1973 on Gala Supreme , who had a Filipino father and a half -
Aboriginal mother . Recent years ( edit ) Wikinews has related news : Delta Blues wins 2006 Melbourne Cup Makybe Diva wins the Melbourne Cup Efficient wins
Melbourne Cup Viewed wins the 2008 Melbourne Cup The race has undergone several alterations in recent years , the most visible being the entry of many foreign -
trained horses . Most have failed to cope with the conditions ; the three successful `` foreign raids '' include two by Irish trainer Dermot K. Weld successful
in 1993 and 2002 , and one in 2006 by Katsumi Yoshida of Japan 's renowned Yoshida racing and breeding family . The attraction for foreigners to compete was ,
primarily , the low - profile change to the new `` quality handicap '' weighting system . The 1910 Melbourne Cup was won by Comedy King , the first foreign bred
horse to do so .
</item 1>

<item 2>
Melbourne Cup
Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 . Martini - Henry , the 1883 Melbourne Cup
winner In 1883 , the hardy New Zealand bred , Martini - Henry won the VRC Derby , the Melbourne Cup and on the following Monday retained his undefeated record
by winning Mares ' Produce Stakes . Phar Lap winning the Melbourne Cup Race from Second Wind and Shadow King on 5 November 1930 . Phar Lap , the most famous
horse in the world of his day , won the 1930 Melbourne Cup at 11 / 8 odds on , the shortest priced favourite in the history of the race . He had to be hidden
away at Geelong before the race after an attempt was made to shoot him and only emerged an hour before the race time of the Cup . Phar Lap also competed in 1929
and 1931 , but came 3rd and 8th respectively , despite heavy favouritism in both years . There are a few legends of the first Aboriginal jockey to ride a
Melbourne Cup .
</item 2>

<item 3>
Melbourne Cup
Perhaps they should have kept their money , however . The outsider Briseis comfortably won by 1 length in the biggest field of all time , and in an extremely
good time , so it is unlikely that the horses who perished could have beaten her . 1877 is also the year that the trainer Etienne de Mestre won his fourth
Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell
( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who ,
though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia
brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth
Melbourne Cup in 1975 .
</item 3>

<item 4>
Melbourne Cup
The race was won by the George Hanlon trained 40 - 1 outsider Piping Lane ridden by John Letts and carrying only 48 kg . 1985 -- The first sponsored Melbourne
Cup , and the first million dollar Cup , with $650,000 for the winner . 1987 -- First female jockey to ride in the cup was Maree Lyndon on Argonaut Style .
Dunaden : 2011 Melbourne Cup winner , painted by Charles Church 1993 -- The Dermot Weld trained Irish gelding Vintage Crop became the first Northern hemisphere
trained horse to win the Melbourne Cup and the first Irish horse . 2001 -- Sheila Laxon , was the first woman trainer to officially win the Melbourne Cup .
However , Mrs. A. McDonald ( 1938 ) with Catalogue was really the first woman trainer to win . Women then could not be registered as trainers in Australia , and
it was her husband who was the registered trainer . Mrs. McDonald 's win was as a female trainer of a female owned horse . 2003 -- First Australian female
jockey to ride in the cup was Clare Lindop on Debben .
</item 4>

<item 5>
Melbourne Cup
1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman (
1989 , 1992 , 1995 , 2004 , 2005 ) Most wins by an owner ( edit ) 6 -- Lloyd Williams ( 1981 , 1985 , 2007 , 2012 , 2016 , 2017 ) 4 -- John Tait ( 1866 , 1868 ,
1871 , 1872 ) 4 -- Etienne de Mestre ( 1861 * , 1862 * , 1867 * , 1878 ) * The winners of these years were leased . 4 -- Dato Tan Chin Nam ( 1974 , 1975 , 1996
, 2008 ) Other records ( edit ) Record winning time -- 3.16.
</item 5>

<item 6>
Melbourne Cup
Maree Lyndon became the first female to ride in the Melbourne Cup , when she partnered Argonaut Style in 1987 , in which she ran second last in the 21 horse
field . In 2004 , Makybe Diva became the first mare to win two cups , and also the first horse to win with different trainers , after David Hall moved to Hong
Kong and transferred her to the Lee Freedman stables . The 2005 Melbourne Cup was held before a crowd of 106,479 . Makybe Diva made history by becoming the only
horse to win the race three times . Trainer Lee Freedman said after the race , `` Go and find the youngest child on the course , because that 's the only person
here who will have a chance of seeing this happen again in their lifetime . '' Due to the 2007 Australian Equine influenza outbreak , believed to have been
started by a horse brought into Australia from Japan , neither Delta Blues nor Pop Rock participated in the 2007 Melbourne Cup . Both horses had been stabled in
Japan .
</item 6>

<item 7>
Melbourne Cup
Subsequent foreign bred horses to win Cup were Backwood 1924 ; Phar Lap 1930 ; Wotan 1936 Belldale Ball 1980 ; At Talaq 1986 ; Kingston Rule 1990 ; Vintage Crop
1993 ; Jeune 1994 ; Media Puzzle 2002 ; Makybe Diva 2003 , 2004 , 2005 ; Americain 2010 and Dunaden 2011 . The 1938 Melbourne Cup was won by trainer Mrs. Allan
McDonald , who conditioned Catalogue . Mrs McDonald was a successful trainer in New Zealand , however , at the time women were not allowed to compete as
trainers in Australia so her husband 's name was officially recorded as the winning trainer . The 2001 edition was won by New Zealand mare Ethereal , trained by
Sheila Laxon , the first woman to formally train a Melbourne Cup winner . She also won the Caulfield Cup , a 2,400 metre race also held in Melbourne , and
therefore has won the `` Cups Double '' .
</item 7>

<item 8>
Melbourne Cup
2003 -- The first Melbourne Cup Tour was conducted around Australia , and the biggest crowd , of 122,736 , is recorded at Flemington . 2005 -- Makybe Diva
became the only horse so far to win the Melbourne Cup three times . 2008 -- The `` Cup King '' , Bart Cummings , took his 12th win in the Melbourne Cup with
Viewed 2010 - 150th anniversary . Americain becomes the first French - trained horse to win the race , and Gerald Mosse the first French jockey . 2011 - French
based horse Dunaden wins from Red Cadeaux in the closest finish ever . 2012 - The first seven finishers in the race were bred in Ireland . 2013 - Trainer Gai
Waterhouse became the first Australian female trainer to train the winner of the race when Fiorente won . Her father , legendary trainer Tommy J. Smith was also
a Melbourne Cup Winning trainer ( Toparoa in 1955 and Just A Dash in 1981 ) . 2014 - Two horses die due to racing in the 2014 Melbourne Cup .
</item 8>

<item 9>
Melbourne Cup
1 Melbourne Cup Tour 3 History 3.1 Early years 3.2 Recent years 3.3 Timeline of notable events 3.4 Public holiday 4 Results and records 4.1 Most wins by a horse
4.2 Most wins by a jockey 4.3 Most wins by a trainer 4.4 Most wins by an owner 4.5 Other records 5 Attendance 6 Off the track 7 See also 8 Notes 9 Further
reading 10 External links Qualifying and race conditions ( edit ) 1932 & 1934 winner Peter Pan ridden by jockey Darby Munro , won twice The race is a quality
handicap for horses 3 years old and over , run over a distance of 3,200 metres , on the first Tuesday in November at Flemington Racecourse . The minimum
handicap weight is 50 kg . There is no maximum weight , but the top allocated weight must not be less than 57 kg . The weight allocated to each horse is
declared by the VRC Handicapper in early September .
</item 9>

<item 10>
Melbourne Cup
Most runners -- 39 ( 1890 ) Fewest runners -- 7 ( 1863 ) Most attempts -- Shadow King made six attempts to win the cup in seven years between 1929 and 1935 . He
ran 6th , 3rd , 2nd , 3rd , 2nd and 4th . Attendance ( edit ) The horse show as well as fashion show of Melbourne Cup takes place on the lawn The event is one
of the most popular spectator events in Australia , with sometimes over 110,000 people , some dressed in traditional formal raceday wear and others in all
manner of exotic and amusing costumes , attending the race . The record crowd was 122,736 in 2003 . The 1926 running of the Cup was the first time the 100,000
mark had been passed . Today the record at Flemington is held by the 2006 Victoria Derby when almost 130,000 attended . In 2007 , a limit was placed on the
Spring Carnival attendance at Flemington Racecourse and race - goers are now required to pre-purchase tickets .
</item 10>
=== END ITEM id=bf3652313595edd3 ===

=== ITEM id=afcece3ab016400d ===
Gold sentences and facts:
Sentence 0: Top 20 rankings as of 16 October 2017 Rank Change Team Points Germany 1631 Brazil 1619 Portugal 1446 Argentina 1445 5 Belgium 1333 6 Poland 1323 7 France 1226 8 Spain 1218 9 Chile 1173 10 Peru 1160 11 Switzerland 1134 12 England 1116 13 Colombia 1095 14 Wales 1072 15 Italy 1066 16 Mexico 1060 17 Uruguay 1034 18 Croatia 1013 19 7 Denmark 1001 20 9 Netherlands 931 * Change from 14 September 2017 Complete rankings at FIFA.com
  Fact 0.0: The top 20 rankings are as of 16 October 2017.
  Fact 0.1: Germany is ranked 1 with 1631 points.
  Fact 0.2: Brazil is ranked 2 with 1619 points.
  Fact 0.3: Portugal is ranked 3 with 1446 points.
  Fact 0.4: Argentina is ranked 4 with 1445 points.
  Fact 0.5: Belgium is ranked 5 with 1333 points.
  Fact 0.6: Poland is ranked 6 with 1323 points.
  Fact 0.7: France is ranked 7 with 1226 points.
  Fact 0.8: Spain is ranked 8 with 1218 points.
  Fact 0.9: Chile is ranked 9 with 1173 points.
  Fact 0.10: Peru is ranked 10 with 1160 points.
  Fact 0.11: Switzerland is ranked 11 with 1134 points.
  Fact 0.12: England is ranked 12 with 1116 points.
  Fact 0.13: Colombia is ranked 13 with 1095 points.
  Fact 0.14: Wales is ranked 14 with 1072 points.
  Fact 0.15: Italy is ranked 15 with 1066 points.
  Fact 0.16: Mexico is ranked 16 with 1060 points.
  Fact 0.17: Uruguay is ranked 17 with 1034 points.
  Fact 0.18: Croatia is ranked 18 with 1013 points.
  Fact 0.19: Denmark is ranked 19 with 1001 points and has a rank change of 7.
  Fact 0.20: Netherlands is ranked 20 with 931 points and has a rank change of 9.
  Fact 0.21: The rank changes are measured from 14 September 2017.
  Fact 0.22: The complete rankings are available at FIFA.com.

Retrieved items:
<item 1>
FIFA World Rankings
However , criticism regarding the unrealistic ranking continued even after the introduction of the new formula . Israel 's climb to 15th in November 2008 raised
a few eyebrows even in the Israeli press , with the team missing a huge opportunity to break into the top 10 after conceding a last gasp equaliser against
Latvia . Similarly , Belgium 's rank of world number 1 in November 2015 was met with skepticism in the press , given that Belgium had only played in one
tournament final stage in the past 13 years . Prior to July 2006 , one of the main criticisms was that the rankings considered the performances of teams over an
eight - year period , and that teams ' ranking positions did not correlate to their recent performances . This criticism has lessened somewhat with the
introduction of a new formula , reflecting results over a four - year period , introduced in July 2006 . However , problems remain , particularly that hosts of
some major tournaments do not take part in qualifying rounds , and instead participate only in friendlies which offer fewer points . This has been a factor
which has resulted in 2014 World Cup hosts Brazil falling to a record low ranking of 22nd in the world .
</item 1>

<item 2>
FIFA World Rankings
2004 , 2005 & 2006 ) 3 ( 2007 , 2009 & 2016 ) 2 ( 1993 & 2001 ) Spain 6 ( 2008 , 2009 , 2010 , 2011 , 2012 & 2013 ) 1 ( 1994 ) 3 ( 2002 , 2003 & 2015 ) Germany
2 ( 1993 & 2014 ) 6 ( 1995 , 1996 , 1997 , 2008 , 2012 & 2013 ) 4 ( 1998 , 2010 , 2011 & 2016 ) Argentina 2 ( 2007 & 2016 ) 3 ( 2001 , 2014 & 2015 ) 5 ( 2000 ,
2004 , 2006 , 2012 & 2013 ) France 1 ( 2001 ) 5 ( 1998 , 2000 , 2002 ,
</item 2>

<item 3>
FIFA World Rankings
Retrieved 13 February 2009 . So the news this week that Israel has risen another place to number 15 in the latest FIFA rankings , above Mexico ( 25 ) , Nigeria
( 22 ) , the US ( 24 ) and Colombia ( 40 ) , only serves to underline the futility of the ranking system . Jump up ^ `` FIFA Ranking : November 2008 preview (
II ) '' . Football-Rankings.info. 20 October 2008 . Retrieved 13 February 2009 . Israel will improve ( again ! ) their best ever ranking , climbing one place to
15th . With a win ( instead of a draw ) against Latvia they would have been tied for 8th place with Russia . Jump up ^ Alistair Tweedale ( 13 October 2015 ) .
`` Belgium rise to No1 in Fifa world rankings after they beat Israel -- despite playing one tournament in 13 years '' . The Telegraph . Retrieved 14 November
2015 . Jump up ^ `` FIFA Rankings '' . Travour.com. 1 June 2006 .
</item 3>

<item 4>
FIFA World Rankings
FIFA World rankings - wikipedia FIFA World rankings This is the latest accepted revision , reviewed on 25 October 2017 . Jump to : navigation , search This
article is about the men 's rankings . For the women 's rankings , see FIFA Women 's World Rankings . Top 20 rankings as of 16 October 2017 Rank Change Team
Points Germany 1631 Brazil 1619 Portugal 1446 Argentina 1445 5 Belgium 1333 6 Poland 1323 7 France 1226 8 Spain 1218 9 Chile 1173 10 Peru 1160 11 Switzerland
1134 12 England 1116 13 Colombia 1095 14 Wales 1072 15 Italy 1066 16 Mexico 1060 17 Uruguay 1034 18 Croatia 1013 19 7 Denmark 1001 20 9 Netherlands 931 * Change
from 14 September 2017 Complete rankings at FIFA.com The FIFA World Ranking is a ranking system for men 's national teams in association football , currently
led by Germany .
</item 4>

<item 5>
FIFA World Rankings
Archived from the original on 25 December 2011 . Retrieved 13 April 2011 . Jump up ^ `` Olympics -- updated draw procedure , with teams '' . CAFOnline.com .
Confederation of African Football . 12 April 2011 . Retrieved 13 April 2011 . ^ Jump up to : `` The world rankings riddle '' . BBC Sport. 21 December 2000 .
Jump up ^ `` FIFA adapting new world rankings '' ( PDF ) . Associated Press . 2 June 2006 . Retrieved 23 July 2017 . Jump up ^ `` FIFA chief praises IFA 's work
'' . The Jerusalem Post . 29 August 2008 . Archived from the original on 27 April 2011 . Retrieved 13 February 2009 . Blatter was also asked about Israel 's
unrealistic FIFA ranking . Despite failing to qualify for a major tournament in 38 years , the national team is currently ranked number 16 in the world , ahead
of such team as Greece , Sweden and Denmark , and just two places below England .
</item 5>

<item 6>
FIFA World Rankings
Travour.com. 1 June 2006 . Archived from the original on 14 June 2006 . Jump up ^ Cummings , Michael . `` FIFA World Rankings Place Brazil 18th , Reinforce
Flawed Nature of System '' . Retrieved 21 May 2013 . Jump up ^ Freitas , Paulo . `` Scolari blames FIFA for Brazil 's poor ranking '' . Retrieved 16 June 2013 .
^ Jump up to : Burnton , Simon ( 17 October 2013 ) . `` With a bit of planning England could have been seeded ahead of the Swiss '' . The Guardian . Retrieved
15 June 2015 . Jump up ^ `` Romania , Wales set to be surprise World Cup top seeds '' . USA Today . Associated Press . 23 July 2015 . Retrieved 29 July 2015 .
Romania took advice from a consultant who decoded some mysteries of a ranking system used by FIFA since 1992 . Jump up ^ Gavril , Victor ( 5 July 2015 ) .
</item 6>

<item 7>
FIFA World Rankings
Rank leaders ( edit ) FIFA World Ranking Leaders When the system was introduced , Germany débuted as the top - ranked team following their extended period of
dominance in which they had reached the three previous FIFA World Cup finals , winning one of them . Brazil took the lead in the run up to the 1994 FIFA World
Cup after winning eight and losing only one of nine qualification matches , while on the way scoring twenty goals and conceding just four . Italy then led for a
short time on the back of their own equally successful World Cup qualifying campaign , after which the top place was re-claimed by Germany . Brazil 's success
in their lengthy qualifying campaign returned them to the lead for a brief period . Germany led again during the 1994 World Cup , until Brazil 's victory in
that competition gave them a large lead that would stand up for nearly seven years , until they were surpassed by a strong France team that captured both the
1998 FIFA World Cup and the 2000 European Football Championship .
</item 7>

<item 8>
FIFA World Rankings
In July 2014 , Germany took over the lead once again , having won the 2014 FIFA World Cup . In July 2015 , Argentina reached the top spot for the first time
since 2008 , after reaching both the 2014 FIFA World Cup Final , as well as the 2015 Copa America Final . In November 2015 , Belgium became the leader in the
FIFA rankings for the first time , after topping their Euro 2016 qualifying group . Belgium led the rankings until April 2016 , when Argentina returned to the
top . On April 6 , 2017 , Brazil returned to the No. 1 spot for the first time since just prior to the 2010 World Cup , but Germany regained the top spot in
July after winning the Confederations Cup . Uses of the rankings ( edit ) The rankings are used by FIFA to rank the progression and current ability of the
national football teams of its member nations , and claims that they create `` a reliable measure for comparing national A-teams '' . They are used as part of
the calculation , or the entire grounds to seed competitions .
</item 8>

<item 9>
FIFA World Rankings
Jump up ^ `` Soccer : Quirky FIFA rankings system sees Israel climb to # 15 '' . The Jerusalem Post . 13 November 2008 . Archived from the original on 16
November 2008 . Retrieved 13 February 2009 . Israel continued its baffling rise up the FIFA rankings on Wednesday , climbing one more position to a best - ever
15th in the world . Amazingly , the national team , which has n't reached a major tournament since the 1970 World Cup , is ahead of such teams as Greece ( 18 )
, Nigeria ( 22 ) , Sweden ( 29 ) , Scotland ( 33 ) , Denmark ( 34 ) and the Republic of Ireland ( 36 ) Jump up ^ `` The Last Word : It 's time to get rid of the
FIFA rankings '' . The Jerusalem Post . 14 November 2008 . Archived from the original on 16 November 2008 . Retrieved 13 February 2009 .
</item 9>

<item 10>
FIFA World Rankings
The teams of the member nations of FIFA , football 's world governing body , are ranked based on their game results with the most successful teams being ranked
highest . The rankings were introduced in December 1992 , and eight teams ( Argentina , Belgium , Brazil , France , Germany , Italy , the Netherlands and Spain
) have held the top position , of which Brazil have spent longest ranked first . A points system is used , with points being awarded based on the results of all
FIFA - recognised full international matches . Under the existing system , rankings are based on a team 's performance over the last four years , with more
recent results and more significant matches being more heavily weighted to help reflect the current competitive state of a team . The ranking system was most
recently revamped after the 2006 World Cup , with the first edition of the new series of rankings issued on 12 July 2006 . The most significant change is that
the rankings are now based on results over the previous four years instead of the previous eight years .
</item 10>
=== END ITEM id=afcece3ab016400d ===

=== ITEM id=82665871df86c039 ===
Gold sentences and facts:
Sentence 0: The 2018 Winter Olympics , officially known as the XXIII Olympic Winter Games ( French : Les XXIIIèmes Jeux olympiques d'hiver ; Hangul : 제 23 회 동계 올림픽 ; RR : Je - isipsamhoe Donggye Ollimpik ) and commonly known as PyeongChang 2018 ( ( phjʌŋ. tɕhaŋ ) ) , is a major international multi-sport event scheduled to take place from 9 to 25 February 2018 in Pyeongchang County , South Korea .
  Fact 0.0: The 2018 Winter Olympics are officially known as the XXIII Olympic Winter Games.
  Fact 0.1: The French name of the 2018 Winter Olympics is Les XXIIIemes Jeux olympiques d'hiver.
  Fact 0.2: The Hangul name of the 2018 Winter Olympics is 제 23 회 동계 올림픽.
  Fact 0.3: The Revised Romanization name of the 2018 Winter Olympics is Je - isipsamhoe Donggye Ollimpik.
  Fact 0.4: The 2018 Winter Olympics are commonly known as PyeongChang 2018.
  Fact 0.5: The 2018 Winter Olympics is a major international multi-sport event.
  Fact 0.6: The 2018 Winter Olympics is scheduled to take place from 9 to 25 February 2018.
  Fact 0.7: The 2018 Winter Olympics is scheduled to take place in Pyeongchang County, South Korea.
Sentence 1: XXIII Olympic Winter Games Host city Pyeongchang , South Korea Motto Passion . Connected . Korean : 하나 된 열정 . ( Hanadoen Yeoljeong ) Nations participating 90 ( estimated ) Events 102 in 7 sports ( 15 disciplines ) Opening ceremony 9 February ( 31 days from now ) Closing ceremony 25 February Stadium Pyeongchang Olympic Stadium Winter < Sochi 2014 Beijing 2022 > Summer < Rio de Janeiro 2016 Tokyo 2020 >
  Fact 1.0: The XXIII Olympic Winter Games have host city Pyeongchang, South Korea.
  Fact 1.1: The motto of the XXIII Olympic Winter Games is Passion. Connected.
  Fact 1.2: The Korean motto of the XXIII Olympic Winter Games is 하나 된 열정 (Hanadoen Yeoljeong).
  Fact 1.3: An estimated 90 nations are participating in the XXIII Olympic Winter Games.
  Fact 1.4: The XXIII Olympic Winter Games have 102 events in 7 sports with 15 disciplines.
  Fact 1.5: The opening ceremony of the XXIII Olympic Winter Games is on 9 February.
  Fact 1.6: The page states the opening ceremony of the XXIII Olympic Winter Games is 31 days from now.
  Fact 1.7: The closing ceremony of the XXIII Olympic Winter Games is on 25 February.
  Fact 1.8: The stadium of the XXIII Olympic Winter Games is Pyeongchang Olympic Stadium.
  Fact 1.9: The navigation box lists Sochi 2014 and Beijing 2022 as Winter Games.
  Fact 1.10: The navigation box lists Rio de Janeiro 2016 and Tokyo 2020 as Summer Games.

Retrieved items:
<item 1>
2018 Winter Olympics
Korean : 하나 된 열정 . ( Hanadoen Yeoljeong ) Nations participating 90 ( estimated ) Events 102 in 7 sports ( 15 disciplines ) Opening ceremony 9 February ( 31 days
from now ) Closing ceremony 25 February Stadium Pyeongchang Olympic Stadium Winter < Sochi 2014 Beijing 2022 > Summer < Rio de Janeiro 2016 Tokyo 2020 >
Pyeongchang Location in South Korea Pyeongchang Winter Olympics Hangul 평창 동계 올림픽 Hanja 平 昌 冬季 올림픽 Revised Romanization Pyeongchang Donggye Ollimpik McCune --
Reischauer P'yŏngch'ang Tonggye Ollimp'ik Part of a series on 2018 Winter Olympics ( show ) Bid process ( bid details ) Venues Torch relay Mascots Broadcasters
Opening ceremony ( flag bearers ) Chronological summary Medal table ( medalists ) World and Olympic records Concerns and controversies Closing ceremony ( flag
bearers ) Paralympics IOC KOC POCOG The 2018 Winter Olympics ,
</item 1>

<item 2>
2018 Winter Olympics
Poland Portugal Puerto Rico Romania San Marino Serbia Singapore Slovakia Slovenia South Africa South Korea ( host ) Spain Sweden Switzerland Chinese Taipei
Thailand Timor - Leste Togo Turkey Ukraine United States Uzbekistan Calendar ( edit ) All dates are KST ( UTC + 9 ) OC Opening ceremony Event competitions Event
finals EG Exhibition gala CC Closing ceremony February 8th Thu 9th Fri 10th Sat 11th Sun 12th Mon 13th Tue 14th Wed 15th Thu 16th Fri 17th Sat 18th Sun 19th Mon
20th Tue 21st Wed 22nd Thu 23rd Fri 24th Sat 25th Sun Events Ceremonies OC CC N / A Alpine skiing 11 Biathlon 11 Bobsleigh Cross country skiing 12 Curling
Figure skating EG 5 Freestyle skiing 10 Ice hockey Luge Nordic combined Short track speed skating 8 Skeleton Ski jumping Snowboarding 10 Speed skating 14 Total
events 5 7 8 8 6 7 5 9 6 5 7 8 6 8 102 Cumulative total 5 12 20 28 34 41 46 55
</item 2>

<item 3>
2018 Winter Olympics
Mexico Peru Puerto Rico United States Asia China Hong Kong India Iran Japan Kazakhstan Kyrgyzstan Lebanon Malaysia Mongolia North Korea Pakistan Philippines
Singapore South Korea Chinese Taipei Thailand Timor - Leste Uzbekistan Europe Albania Andorra Armenia Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina
Bulgaria Croatia Cyprus Czech Republic Denmark Estonia Finland France Georgia Germany Great Britain Greece Hungary Iceland Ireland Israel Italy Kosovo Latvia
Liechtenstein Lithuania Luxembourg Macedonia Malta Moldova Monaco Montenegro Netherlands Norway Poland Portugal Romania San Marino Serbia Slovakia Slovenia
Spain Sweden Switzerland Turkey Ukraine Oceania Australia New Zealand Other Olympic Athletes from Russia Venues of the 2018 Winter Olympics Pyeongchang Mountain
Cluster Alpensia Biathlon Centre Cross-Country Centre Ski Jumping Centre Convention Centre IBC Alpensia Sliding Centre Yongpyong Alpine Centre Pyeongchang
Olympic Stadium Pyeongchang Olympic Village Bokwang Phoenix Park Stand - alone venues : Jeongseon Alpine Centre Phoenix Snow Park Gangneung Coastal Cluster
Gangneung Olympic Park Gangneung Curling Centre Gangneung Hockey Centre Gangneung Ice Arena Gangneung Oval Kwandong Hockey Centre Gangneung Olympic Village
Italics indicate non-competition venues .
</item 3>

<item 4>
2018 Winter Olympics
In the United States , the Games will once again be broadcast by NBC under a long - term contract with NBCUniversal ; it will be NBC 's first Olympics without
long - time primary host Bob Costas , who announced on 7 February 2017 his retirement from the role in favour of Mike Tirico . On 28 March 2017 , NBC also
announced that it would air most primetime coverage simultaneously in all time zones in the United States , rather than tape - delayed for the West Coast .
Concerns and controversies ( edit ) Main article : Concerns and controversies at the 2018 Winter Olympics Security ( edit ) On 20 September 2017 , South Korea
's President Moon Jae - in said the country is pushing to ensure security at the Pyeongchang Winter Olympic Games amid rising tensions over North Korea 's
nuclear tests and a series of missile launches . However , on the next day , French Minister of Youth Affairs and Sports Laura Flessel - Colovic said France 's
Winter Olympics team will boycott the games unless South Korea 's security is guaranteed . On 22 September 2017 , Austria and Germany joined France in
considering not attending the Games .
</item 4>

<item 5>
2018 Winter Olympics
Louis 1908 London 1912 Stockholm 1916 Berlin 1920 Antwerp 1924 Paris 1928 Amsterdam 1932 Los Angeles 1936 Berlin 1940 Tokyo 1944 London 1948 London 1952
Helsinki 1956 Melbourne 1960 Rome 1964 Tokyo 1968 Mexico City 1972 Munich 1976 Montreal 1980 Moscow 1984 Los Angeles 1988 Seoul 1992 Barcelona 1996 Atlanta 2000
Sydney 2004 Athens 2008 Beijing 2012 London 2016 Rio de Janeiro 2020 Tokyo 2024 Paris 2028 Los Angeles 2032 TBD Winter Games 1924 Chamonix 1928 St. Moritz 1932
Lake Placid 1936 Garmisch - Partenkirchen 1940 Sapporo 1944 Cortina d'Ampezzo 1948 St.
</item 5>

<item 6>
2018 Winter Olympics
1 Official sanctions 10.3. 2 Reaction in Russia 10.3. 3 Criticism 11 See also 12 Notes and references 13 External links Bidding ( edit ) Main article : Bids for
the 2018 Winter Olympics Pyeongchang bid to host both the 2010 and 2014 Winter Olympic Games but lost in the final rounds of voting by three and four votes
respectively . Pyeongchang won its bid for the 2018 Winter Olympic Games in the first round of voting , receiving 63 of the 95 votes cast , giving it the
majority required to be elected host city . Munich also launched a bid to host these Games . Prior to Beijing 's successful 2022 Winter Olympics bid , Munich
would have become the first city to host both the Winter and the Summer Games , having previously hosted the 1972 Summer Olympics , but received 25 votes .
Annecy ( in southeastern France ) launched a bid , but failed to secure public support from local citizens . Their bid received seven votes .
</item 6>

<item 7>
2018 Winter Olympics
Pyeongchang will be the third Asian city to host the Winter Games ; the first two were in Japan , at Sapporo ( 1972 ) and Nagano ( 1998 ) . Contents ( hide ) 1
Bidding 1.1 Host city election 2 Tickets 3 Preparations 3.1 Torch relay 4 Venues 4.1 Pyeongchang ( Mountain cluster ) 4.1. 1 Alpensia Sports Park 4.1. 2 Stand -
alone venues 4.2 Gangneung ( Coastal cluster ) 5 Sports 6 Participating National Olympic Committees 7 Calendar 8 Marketing 8.1 Branding 8.2 Mascots 9
Broadcasting rights 10 Concerns and controversies 10.1 Security 10.2 Jeongseon Alpine Centre Ecological Issues 10.3 Russian doping 10.3. 1 Official sanctions
10.3. 2 Reaction in Russia 10.3.
</item 7>

<item 8>
2018 Winter Olympics
Moritz 1952 Oslo 1956 Cortina d'Ampezzo 1960 Squaw Valley 1964 Innsbruck 1968 Grenoble 1972 Sapporo 1976 Innsbruck 1980 Lake Placid 1984 Sarajevo 1988 Calgary
1992 Albertville 1994 Lillehammer 1998 Nagano 2002 Salt Lake City 2006 Turin 2010 Vancouver 2014 Sochi 2018 Pyeongchang 2022 Beijing 2026 TBD 2030 TBD Ancient
Olympic Games Intercalated Games 1906 Paralympic Games Youth Olympic Games Events at the 2018 Winter Olympics ( PyeongChang , South Korea ) Alpine skiing
Biathlon Bobsleigh Cross-country skiing Curling Figure skating Freestyle skiing Ice hockey men women Luge Nordic combined Short track speed skating Skeleton Ski
jumping Snowboarding Speed skating Nations at the 2018 Winter Olympics in Pyeongchang , South Korea Africa Eritrea Kenya Madagascar Morocco South Africa Togo
Americas Argentina Bermuda Bolivia Brazil Canada Chile Colombia Dominica Ecuador Mexico Peru Puerto Rico United States Asia China Hong Kong India Iran Japan
Kazakhstan Kyrgyzstan Lebanon Malaysia Mongolia
</item 8>

<item 9>
2018 Winter Olympics
officially known as the XXIII Olympic Winter Games ( French : Les XXIIIèmes Jeux olympiques d'hiver ; Hangul : 제 23 회 동계 올림픽 ; RR : Je - isipsamhoe Donggye
Ollimpik ) and commonly known as PyeongChang 2018 ( ( phjʌŋ. tɕhaŋ ) ) , is a major international multi-sport event scheduled to take place from 9 to 25
February 2018 in Pyeongchang County , South Korea . The elected host city was announced on 6 July 2011 by the International Olympic Committee ( IOC ) after the
123rd IOC Session in Durban , South Africa . Pyeongchang won its bid on the first round of voting , receiving more votes than both Munich , Germany and Annecy ,
France combined . These will be South Korea 's second Olympic Games and its first Winter Games ; Seoul hosted the Summer Games in 1988 .
</item 9>

<item 10>
2018 Winter Olympics
The New York Times . Retrieved 9 January 2018 . Jump up ^ `` With one year until 2018 Winter Games , Russia 's status murky '' . 9 February 2017 . Jump up ^ ``
IOC suspends Russian NOC and creates a path for clean individual athletes to compete in Pyeongchang 2018 under the Olympic flag '' . IOC. 5 December 2017 .
Retrieved 5 December 2017 . Jump up ^ `` Quota allocation for Alpine skiing '' . www.data.fis-ski.com/ . International Ski Federation ( FIS ) . 8 August 2017 .
Retrieved 8 August 2017 . Jump up ^ `` 2018 Winter Olympics '' . IIHF . iihf.com . Retrieved 18 May 2015 . Jump up ^ `` Quota allocation for Cross-country
skiing '' . www.data.fis-ski.com/ . International Ski Federation ( FIS ) . 8 August 2017 . Retrieved 8 August 2017 . Jump up ^ `` Pyeongchang 2018 Winter
Olympics '' .
</item 10>
=== END ITEM id=82665871df86c039 ===

=== ITEM id=bf0dc65b758d9659 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 70 `` The Flash Reborn '' Glen Winter Story by : Andrew Kreisberg Teleplay by : Todd Helbing & Eric Wallace October 10 , 2017 ( 2017 - 10 - 10 ) T27. 13401 2.84 71 `` Mixed Signals '' Alexandra La Roche Jonathan Butler & Gabriel Garza October 17 , 2017 ( 2017 - 10 - 17 ) T27. 13402 2.54 72 `` Luck Be a Lady '' Armen V. Kevorkian Sam Chalsen & Judalina Neira October 24 , 2017 ( 2017 - 10 - 24 ) T27. 13403 2.62 73 `` Elongated Journey Into Night '' Tom Cavanagh Sterling Gates & Thomas Pound October 31 , 2017 ( 2017 - 10 - 31 ) T27. 13404 1.99 74 5 `` Girls Night Out '' Laura Belsey Lauren Certo & Kristen Kim November 7 , 2017 ( 2017 - 11 - 07 ) T27. 13405 2.38 75 6 `` When Harry Met Harry ... '' Brent Crowell Jonathan Butler & Gabriel Garza November 14 , 2017 ( 2017 - 11 - 14 ) T27. 13406 2.46 76 7 `` Therefore I Am '' David McWhirter Eric Wallace & Thomas Pound November 21 , 2017 ( 2017 - 11 - 21 ) T27. 13407 2.20 77 8 `` Crisis on Earth - X , Part 3 '' Dermott Downs Story by : Andrew Kreisberg & Marc Guggenheim Teleplay by : Todd Helbing November 28 , 2017 ( 2017 - 11 - 28 ) T27. 13408 2.82 78 9 `` Do n't Run '' Stefan Pleszczynski Sam Chalsen & Judalina Neira December 5 , 2017 ( 2017 - 12 - 05 ) T27. 13409 2.22 79 10 `` The Trial of The Flash '' Philip Chipera Lauren Certo & Kristen Kim January 16 , 2018 ( 2018 - 01 - 16 ) T27. 13410 2.51 80 11 `` The Elongated Knight Rises '' Alexandra La Roche Sterling Gates & Thomas Pound January 23 , 2018 ( 2018 - 01 - 23 ) T27. 13411 2.12 81 12 `` Honey , I Shrunk Team Flash '' Chris Peppe Sam Chalsen & Judalina Neira January 30 , 2018 ( 2018 - 01 - 30 ) T27. 13412 2.60 82 13 `` True Colors '' Tara Nicole Weyr Jonathan Butler & Gabriel Garza February 6 , 2018 ( 2018 - 02 - 06 ) T27. 13413 2.28 83 14 `` Subject 9 '' Ralph Hemecker Mike Alber & Gabe Snyder February 27 , 2018 ( 2018 - 02 - 27 ) T27. 13414 2.12 84 15 `` Enter Flashtime '' Gregory Smith Todd Helbing & Sterling Gates March 6 , 2018 ( 2018 - 03 - 06 ) T27. 13415 2.04 85 16 `` Run , Iris , Run '' Harry Jierjian Eric Wallace March 13 , 2018 ( 2018 - 03 - 13 ) T27. 13416 2.09 86 17 `` Null and Annoyed '' Kevin Smith Lauren Certo & Kristen Kim April 10 , 2018 ( 2018 - 04 - 10 ) T27. 13417 1.82 87 18 `` Lose Yourself '' Hanelle Culpepper Jonathan Butler & Gabriel Garza April 17 , 2018 ( 2018 - 04 - 17 ) T27. 13418 1.88 88 19 `` Fury Rogue '' Rachel Talalay Joshua V. Gilbert & Jeff Hersh April 24 , 2018 ( 2018 - 04 - 24 ) TBA 1.90 89 20 `` Therefore She Is '' Rob J. Greenlea Sterling Gates & Thomas Pound May 1 , 2018 ( 2018 - 05 - 01 ) TBA TBD 90 21 `` Harry and the Harrisons '' Kevin Mock Judalina Neira & Lauren Certo May 8 , 2018 ( 2018 - 05 - 08 ) TBA TBD 91 22 `` Think Fast '' Viet Nguyen Sam Chalsen & Kristen Kim May 15 , 2018 ( 2018 - 05 - 15 ) TBA TBD 92 23 `` We Are the Flash '' TBA TBA May 22 , 2018 ( 2018 - 05 - 22 ) TBA TBD
  Fact 0.0: Episode 70 overall is titled The Flash Reborn.
  Fact 0.1: The Flash Reborn was directed by Glen Winter.
  Fact 0.2: The Flash Reborn had story by Andrew Kreisberg and teleplay by Todd Helbing and Eric Wallace.
  Fact 0.3: The Flash Reborn aired October 10, 2017, with production code T27.13401 and 2.84 million U.S. viewers.
  Fact 0.4: Episode 71 overall is titled Mixed Signals.
  Fact 0.5: Mixed Signals was directed by Alexandra La Roche.
  Fact 0.6: Mixed Signals was written by Jonathan Butler and Gabriel Garza.
  Fact 0.7: Mixed Signals aired October 17, 2017, with production code T27.13402 and 2.54 million U.S. viewers.
  Fact 0.8: Episode 72 overall is titled Luck Be a Lady.
  Fact 0.9: Luck Be a Lady was directed by Armen V. Kevorkian.
  Fact 0.10: Luck Be a Lady was written by Sam Chalsen and Judalina Neira.
  Fact 0.11: Luck Be a Lady aired October 24, 2017, with production code T27.13403 and 2.62 million U.S. viewers.
  Fact 0.12: Episode 73 overall is titled Elongated Journey Into Night.
  Fact 0.13: Elongated Journey Into Night was directed by Tom Cavanagh.
  Fact 0.14: Elongated Journey Into Night was written by Sterling Gates and Thomas Pound.
  Fact 0.15: Elongated Journey Into Night aired October 31, 2017, with production code T27.13404 and 1.99 million U.S. viewers.
  Fact 0.16: Episode 74 overall is episode 5 of its season and is titled Girls Night Out.
  Fact 0.17: Girls Night Out was directed by Laura Belsey.
  Fact 0.18: Girls Night Out was written by Lauren Certo and Kristen Kim.
  Fact 0.19: Girls Night Out aired November 7, 2017, with production code T27.13405 and 2.38 million U.S. viewers.
  Fact 0.20: Episode 75 overall is episode 6 of its season and is titled When Harry Met Harry ...
  Fact 0.21: When Harry Met Harry ... was directed by Brent Crowell.
  Fact 0.22: When Harry Met Harry ... was written by Jonathan Butler and Gabriel Garza.
  Fact 0.23: When Harry Met Harry ... aired November 14, 2017, with production code T27.13406 and 2.46 million U.S. viewers.
  Fact 0.24: Episode 76 overall is episode 7 of its season and is titled Therefore I Am.
  Fact 0.25: Therefore I Am was directed by David McWhirter.
  Fact 0.26: Therefore I Am was written by Eric Wallace and Thomas Pound.
  Fact 0.27: Therefore I Am aired November 21, 2017, with production code T27.13407 and 2.20 million U.S. viewers.
  Fact 0.28: Episode 77 overall is episode 8 of its season and is titled Crisis on Earth - X, Part 3.
  Fact 0.29: Crisis on Earth - X, Part 3 was directed by Dermott Downs.
  Fact 0.30: Crisis on Earth - X, Part 3 had story by Andrew Kreisberg and Marc Guggenheim and teleplay by Todd Helbing.
  Fact 0.31: Crisis on Earth - X, Part 3 aired November 28, 2017, with production code T27.13408 and 2.82 million U.S. viewers.
  Fact 0.32: Episode 78 overall is episode 9 of its season and is titled Don't Run.
  Fact 0.33: Don't Run was directed by Stefan Pleszczynski.
  Fact 0.34: Don't Run was written by Sam Chalsen and Judalina Neira.
  Fact 0.35: Don't Run aired December 5, 2017, with production code T27.13409 and 2.22 million U.S. viewers.
  Fact 0.36: Episode 79 overall is episode 10 of its season and is titled The Trial of The Flash.
  Fact 0.37: The Trial of The Flash was directed by Philip Chipera.
  Fact 0.38: The Trial of The Flash was written by Lauren Certo and Kristen Kim.
  Fact 0.39: The Trial of The Flash aired January 16, 2018, with production code T27.13410 and 2.51 million U.S. viewers.
  Fact 0.40: Episode 80 overall is episode 11 of its season and is titled The Elongated Knight Rises.
  Fact 0.41: The Elongated Knight Rises was directed by Alexandra La Roche.
  Fact 0.42: The Elongated Knight Rises was written by Sterling Gates and Thomas Pound.
  Fact 0.43: The Elongated Knight Rises aired January 23, 2018, with production code T27.13411 and 2.12 million U.S. viewers.
  Fact 0.44: Episode 81 overall is episode 12 of its season and is titled Honey, I Shrunk Team Flash.
  Fact 0.45: Honey, I Shrunk Team Flash was directed by Chris Peppe.
  Fact 0.46: Honey, I Shrunk Team Flash was written by Sam Chalsen and Judalina Neira.
  Fact 0.47: Honey, I Shrunk Team Flash aired January 30, 2018, with production code T27.13412 and 2.60 million U.S. viewers.
  Fact 0.48: Episode 82 overall is episode 13 of its season and is titled True Colors.
  Fact 0.49: True Colors was directed by Tara Nicole Weyr.
  Fact 0.50: True Colors was written by Jonathan Butler and Gabriel Garza.
  Fact 0.51: True Colors aired February 6, 2018, with production code T27.13413 and 2.28 million U.S. viewers.
  Fact 0.52: Episode 83 overall is episode 14 of its season and is titled Subject 9.
  Fact 0.53: Subject 9 was directed by Ralph Hemecker.
  Fact 0.54: Subject 9 was written by Mike Alber and Gabe Snyder.
  Fact 0.55: Subject 9 aired February 27, 2018, with production code T27.13414 and 2.12 million U.S. viewers.
  Fact 0.56: Episode 84 overall is episode 15 of its season and is titled Enter Flashtime.
  Fact 0.57: Enter Flashtime was directed by Gregory Smith.
  Fact 0.58: Enter Flashtime was written by Todd Helbing and Sterling Gates.
  Fact 0.59: Enter Flashtime aired March 6, 2018, with production code T27.13415 and 2.04 million U.S. viewers.
  Fact 0.60: Episode 85 overall is episode 16 of its season and is titled Run, Iris, Run.
  Fact 0.61: Run, Iris, Run was directed by Harry Jierjian.
  Fact 0.62: Run, Iris, Run was written by Eric Wallace.
  Fact 0.63: Run, Iris, Run aired March 13, 2018, with production code T27.13416 and 2.09 million U.S. viewers.
  Fact 0.64: Episode 86 overall is episode 17 of its season and is titled Null and Annoyed.
  Fact 0.65: Null and Annoyed was directed by Kevin Smith.
  Fact 0.66: Null and Annoyed was written by Lauren Certo and Kristen Kim.
  Fact 0.67: Null and Annoyed aired April 10, 2018, with production code T27.13417 and 1.82 million U.S. viewers.
  Fact 0.68: Episode 87 overall is episode 18 of its season and is titled Lose Yourself.
  Fact 0.69: Lose Yourself was directed by Hanelle Culpepper.
  Fact 0.70: Lose Yourself was written by Jonathan Butler and Gabriel Garza.
  Fact 0.71: Lose Yourself aired April 17, 2018, with production code T27.13418 and 1.88 million U.S. viewers.
  Fact 0.72: Episode 88 overall is episode 19 of its season and is titled Fury Rogue.
  Fact 0.73: Fury Rogue was directed by Rachel Talalay.
  Fact 0.74: Fury Rogue was written by Joshua V. Gilbert and Jeff Hersh.
  Fact 0.75: Fury Rogue aired April 24, 2018, with production code TBA and 1.90 million U.S. viewers.
  Fact 0.76: Episode 89 overall is episode 20 of its season and is titled Therefore She Is.
  Fact 0.77: Therefore She Is was directed by Rob J. Greenlea.
  Fact 0.78: Therefore She Is was written by Sterling Gates and Thomas Pound.
  Fact 0.79: Therefore She Is aired May 1, 2018, with production code TBA and viewer count TBD.
  Fact 0.80: Episode 90 overall is episode 21 of its season and is titled Harry and the Harrisons.
  Fact 0.81: Harry and the Harrisons was directed by Kevin Mock.
  Fact 0.82: Harry and the Harrisons was written by Judalina Neira and Lauren Certo.
  Fact 0.83: Harry and the Harrisons aired May 8, 2018, with production code TBA and viewer count TBD.
  Fact 0.84: Episode 91 overall is episode 22 of its season and is titled Think Fast.
  Fact 0.85: Think Fast was directed by Viet Nguyen.
  Fact 0.86: Think Fast was written by Sam Chalsen and Kristen Kim.
  Fact 0.87: Think Fast aired May 15, 2018, with production code TBA and viewer count TBD.
  Fact 0.88: Episode 92 overall is episode 23 of its season and is titled We Are the Flash.
  Fact 0.89: The director and writers of We Are the Flash are listed as TBA.
  Fact 0.90: We Are the Flash aired May 22, 2018, with production code TBA and viewer count TBD.

Retrieved items:
<item 1>
List of The Flash episodes
13418 1.88 88 19 `` Fury Rogue '' Rachel Talalay Joshua V. Gilbert & Jeff Hersh April 24 , 2018 ( 2018 - 04 - 24 ) TBA 1.90 89 20 `` Therefore She Is '' Rob J.
Greenlea Sterling Gates & Thomas Pound May 1 , 2018 ( 2018 - 05 - 01 ) TBA TBD 90 21 `` Harry and the Harrisons '' Kevin Mock Judalina Neira & Lauren Certo May
8 , 2018 ( 2018 - 05 - 08 ) TBA TBD 91 22 `` Think Fast '' Viet Nguyen Sam Chalsen & Kristen Kim May 15 , 2018 ( 2018 - 05 - 15 ) TBA TBD 92 23 `` We Are the
Flash '' TBA TBA May 22 , 2018 ( 2018 - 05 - 22 ) TBA TBD Ratings ( edit ) Season 1 ( 2014 -- 15 ) ( edit ) No .
</item 1>

<item 2>
List of The Flash episodes
13415 2.04 85 16 `` Run , Iris , Run '' Harry Jierjian Eric Wallace March 13 , 2018 ( 2018 - 03 - 13 ) T27. 13416 2.09 86 17 `` Null and Annoyed '' Kevin Smith
Lauren Certo & Kristen Kim April 10 , 2018 ( 2018 - 04 - 10 ) T27. 13417 1.82 87 18 `` Lose Yourself '' Hanelle Culpepper Jonathan Butler & Gabriel Garza April
17 , 2018 ( 2018 - 04 - 17 ) T27.
</item 2>

<item 3>
List of The Flash episodes
2 / 4 3.14 0.8 1.80 2.0 4.94 10 `` Borrowing Problems From the Future '' January 24 , 2017 1.0 / 3 2.72 0.8 1.92 1.8 4.64 11 `` Dead or Alive '' January 31 ,
2017 1.1 / 4 3.06 0.7 1.64 1.8 4.71 12 `` Untouchable '' February 7 , 2017 1.1 / 4 2.91 0.7 1.68 1.8 4.59 13 `` Attack on Gorilla City '' February 21 , 2017 1.0
/ 4 2.78 0.9 1.96 1.9 4.75 14 `` Attack on Central City '' February 28 , 2017 1.1 / 4 2.87 0.8 1.81 1.9 4.68 15 `` The Wrath of Savitar '' March 7 ,
</item 3>

<item 4>
List of The Flash episodes
13406 2.46 76 7 `` Therefore I Am '' David McWhirter Eric Wallace & Thomas Pound November 21 , 2017 ( 2017 - 11 - 21 ) T27. 13407 2.20 77 8 `` Crisis on Earth -
X , Part 3 '' Dermott Downs Story by : Andrew Kreisberg & Marc Guggenheim Teleplay by : Todd Helbing November 28 , 2017 ( 2017 - 11 - 28 ) T27. 13408 2.82 78 9
`` Do n't Run '' Stefan Pleszczynski Sam Chalsen & Judalina Neira December 5 , 2017 ( 2017 - 12 - 05 ) T27. 13409 2.22 79 10 `` The Trial of The Flash '' Philip
Chipera Lauren Certo & Kristen Kim January 16 , 2018 ( 2018 - 01 - 16 ) T27.
</item 4>

<item 5>
List of The Flash episodes
Retrieved February 11 , 2017 . ^ Jump up to : Porter , Rick ( February 1 , 2017 ) . `` ' The Wall ' adjusts up , ' Bones ' adjusts down : Tuesday final ratings
'' . TV by the Numbers . Archived from the original on February 2 , 2017 . Retrieved February 1 , 2017 . ^ Jump up to : Porter , Rick ( February 8 , 2017 ) . ``
' The Middle ' , ' The Wall ' , ' This Is Us ' , FOX comedies all adjust up : Tuesday final ratings '' . TV by the Numbers . Archived from the original on
February 8 , 2017 . Retrieved February 8 , 2017 . ^ Jump up to : Porter , Rick ( February 23 , 2017 ) . `` ' The Flash ' , ' NCIS ' , all others unchanged :
Tuesday final ratings '' . TV by the Numbers . Archived from the original on February 23 , 2017 . Retrieved February 23 , 2017 .
</item 5>

<item 6>
List of The Flash episodes
13401 2.84 71 `` Mixed Signals '' Alexandra La Roche Jonathan Butler & Gabriel Garza October 17 , 2017 ( 2017 - 10 - 17 ) T27. 13402 2.54 72 `` Luck Be a Lady
'' Armen V. Kevorkian Sam Chalsen & Judalina Neira October 24 , 2017 ( 2017 - 10 - 24 ) T27. 13403 2.62 73 `` Elongated Journey Into Night '' Tom Cavanagh
Sterling Gates & Thomas Pound October 31 , 2017 ( 2017 - 10 - 31 ) T27. 13404 1.99 74 5 `` Girls Night Out '' Laura Belsey Lauren Certo & Kristen Kim November 7
, 2017 ( 2017 - 11 - 07 ) T27. 13405 2.38 75 6 `` When Harry Met Harry ... '' Brent Crowell Jonathan Butler & Gabriel Garza November 14 , 2017 ( 2017 - 11 - 14
) T27.
</item 6>

<item 7>
List of The Flash episodes
9 4.68 15 `` The Wrath of Savitar '' March 7 , 2017 0.9 / 3 2.52 0.7 1.71 1.6 4.23 16 `` Into the Speed Force '' March 14 , 2017 0.9 / 3 2.39 0.7 1.70 1.6 4.10
17 `` Duet '' March 21 , 2017 1.0 / 4 2.71 0.8 1.88 1.8 4.59 18 `` Abra Kadabra '' March 28 , 2017 0.9 / 3 2.39 0.7 1.67 1.6 4.06 19 `` The Once and Future
Flash '' April 25 , 2017 1.0 / 4 2.67 0.9 1.90 1.9 4.57 20 `` I Know Who You Are '' May 2 , 2017 1.0 / 4 2.69 0.8 1.
</item 7>

<item 8>
List of The Flash episodes
13410 2.51 80 11 `` The Elongated Knight Rises '' Alexandra La Roche Sterling Gates & Thomas Pound January 23 , 2018 ( 2018 - 01 - 23 ) T27. 13411 2.12 81 12 ``
Honey , I Shrunk Team Flash '' Chris Peppe Sam Chalsen & Judalina Neira January 30 , 2018 ( 2018 - 01 - 30 ) T27. 13412 2.60 82 13 `` True Colors '' Tara Nicole
Weyr Jonathan Butler & Gabriel Garza February 6 , 2018 ( 2018 - 02 - 06 ) T27. 13413 2.28 83 14 `` Subject 9 '' Ralph Hemecker Mike Alber & Gabe Snyder February
27 , 2018 ( 2018 - 02 - 27 ) T27. 13414 2.12 84 15 `` Enter Flashtime '' Gregory Smith Todd Helbing & Sterling Gates March 6 , 2018 ( 2018 - 03 - 06 ) T27.
</item 8>

<item 9>
List of The Flash episodes
25 23 October 4 , 2016 ( 2016 - 10 - 04 ) May 23 , 2017 ( 2017 - 05 - 23 ) 120 3.50 23 October 10 , 2017 ( 2017 - 10 - 10 ) May 22 , 2018 ( 2018 - 05 - 22 ) TBA
TBA Episodes ( edit ) Season 1 ( 2014 -- 15 ) ( edit ) No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers (
millions ) `` Pilot '' David Nutter Story by : Greg Berlanti & Andrew Kreisberg & Geoff Johns Teleplay by : Andrew Kreisberg & Geoff Johns October 7 , 2014 (
2014 - 10 - 07 ) 296648 4.83 Barry Allen , a forensic crime scene assistant for Central City Police Department , is struck by lightning during a storm that is
created after a particle accelerator from S.T.A.R. Labs explodes . In a coma for nine months , Barry awakens in S.T.A.R.
</item 9>

<item 10>
List of The Flash episodes
^ Jump up to : Bibel , Sara ( October 16 , 2014 ) . `` Revised Tuesday , October 7 Final Ratings : ' Selfie ' , ' Marvel 's Agents of S.H.I.E.L.D. ' & ' Forever
' Adjusted Down '' . TV by the Numbers . Archived from the original on October 18 , 2014 . Retrieved October 16 , 2014 . ^ Jump up to : Bibel , Sara ( October
15 , 2014 ) . `` Tuesday Final Ratings : ' The Flash ' , ' Marvel 's Agents of S.H.I.E. LD . ' , ' The Voice ' & ' The Mindy Project ' Adjusted Up ; ' About A
Boy ' & ' Chicago Fire ' Adjusted Down '' . TV by the Numbers . Archived from the original on October 16 , 2014 . Retrieved October 15 , 2014 . ^ Jump up to :
Kondolojy , Amanda ( October 22 , 2014 ) .
</item 10>
=== END ITEM id=bf0dc65b758d9659 ===

=== ITEM id=687a59c4e36c2234 ===
Gold sentences and facts:
Sentence 0: `` What a Friend We Have in Jesus '' is a Christian hymn originally written by Joseph M. Scriven as a poem in 1855 to comfort his mother who was living in Ireland while he was in Canada . Scriven originally published the poem anonymously , and only received full credit for it in the 1880s . The tune to the hymn was composed by Charles Crozat Converse in 1868 . William Bolcom composed a setting of the hymn .
  Fact 0.0: "What a Friend We Have in Jesus" is a Christian hymn.
  Fact 0.1: The hymn "What a Friend We Have in Jesus" was originally written by Joseph M. Scriven as a poem in 1855.
  Fact 0.2: Joseph M. Scriven wrote the poem to comfort his mother.
  Fact 0.3: Joseph M. Scriven's mother was living in Ireland while he was in Canada.
  Fact 0.4: Joseph M. Scriven originally published the poem anonymously.
  Fact 0.5: Joseph M. Scriven only received full credit for the poem in the 1880s.
  Fact 0.6: The tune of the hymn "What a Friend We Have in Jesus" was composed by Charles Crozat Converse in 1868.
  Fact 0.7: William Bolcom composed a setting of the hymn "What a Friend We Have in Jesus".

Retrieved items:
<item 1>
What a Friend We Have in Jesus
In some settings , the lyrics have been matched to other tunes such as the Welsh `` Calon Lân '' ( originally wedded to the Welsh poem translated as `` A Pure
Heart '' ) . What a Friend We Have In Jesus Performed by the Onward Brass Band Problems playing this file ? See media help . Contents ( hide ) 1 Renditions 1.1
In Asia 2 In popular culture 3 Published versions 4 References 5 External links Renditions ( edit ) Washington Phillips , as `` Jesus Is My Friend '' ( 1928 ,
Columbia Records ) Bing Crosby ( 1951 , Beloved Hymns ) Rosemary Clooney ( 1959 , Hymns from the Heart , MGM Records ) Ella Fitzgerald with the Ralph Carmichael
Choir ( 1967 , Brighten the Corner ) Aretha Franklin ( 1972 , Amazing Grace ) Alan Price used the tune for his song `` Changes '' in the 1973 film O Lucky Man !
, which was reused in a Volkswagen commercial of the same name with Paula Hamilton in the 1980s .
</item 1>

<item 2>
What a Friend We Have in Jesus
External links ( edit ) Piano score ( pdf ) at Timeless Truths Text , MIDI , and piano score for `` What a Friend We Have in Jesus '' at HymnSite.
</item 2>

<item 3>
What a Friend We Have in Jesus
wikipedia.org/w/index.php?title=What_a_Friend_We_Have_in_Jesus&oldid=831327264 '' Categories : 1928 songs Canadian Christian hymns Columbia Records singles Glen
Campbell songs Mississippi John Hurt songs Pace Jubilee Singers songs Songs about Jesus Washington Phillips songs Hidden categories : Articles with hAudio
microformats Articles containing Japanese - language text Articles with trivia sections from February 2018 All articles with trivia sections Talk Contents About
Wikipedia Wikisource 日本 語 Norsk Svenska 中文 Edit links This page was last edited on 20 March 2018 , at 00 : 54 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
What a Friend We Have in Jesus
`` Merciful '' ) , which is what it is best known by . In 1910 , Daisui Sugitani re-wrote the lyrics in Japanese and changed the title to `` Hoshi no Yo '' ( 星
の 界 , lit . `` World of Stars '' ) . Another version was written by Ryūkō Kawaji with the title `` Hoshi no Sekai '' ( 星 の 世界 , lit . `` World of Stars '' ) .
It is also known by the title `` Tsumitoga o Ninou '' ( つみ と が を に の う , lit . `` To Bear Sin '' ) . The hymn is popular at wedding ceremonies in Japan . In
Indonesia , the hymn is known as `` Yesus Kawan yang Sejati '' and is sung in Indonesian or Batak ( the indigenous language of North Sumatra ) in Manado ,
Maluccan , and Protestant churches ( around 6 % of the population ) .
</item 4>

<item 5>
What a Friend We Have in Jesus
Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic . Retrieved August 24 , 2015 . Jump up ^ `` Volkswagen Golf -- Paula Hamilton -- 1987 --
UK Advert '' . Retro Commercials . March 13 , 2010 . Retrieved September 26 , 2011 . Jump up ^ `` Pure Hymns overview '' . Allmusic . Retrieved July 4 , 2010 .
Jump up ^ `` Monty Alexander UPLIFT 2 '' . Jazz Legacy Productions . Retrieved January 11 , 2014 . Jump up ^ `` Itsukushimi Fukaki song sheet '' ( PDF ) .
Retrieved 2007 - 07 - 16 . ^ Jump up to : `` 312 : What a Friend We Have in Jesus '' . Retrieved 2008 - 03 - 19 . Jump up ^ Christhiya Keerthanangal ( Christian
Hymns ) . 2007 . Hymn 173 . External links ( edit ) Piano score ( pdf ) at Timeless Truths Text , MIDI ,
</item 5>

<item 6>
What a Friend We Have in Jesus
The tune was in the labor protest movement due to its wide popularity . `` Dump the Bosses Off Your Backs '' ( words by John Brill ) was published in the IWW ``
Little Red Songbook '' and has been recorded by dozens of artists including Utah Phillips , Anne Feeney , and Joe Glazer . A Norwegian version was used in the
film Trollhunter . Published versions ( edit ) Rise Up Singing , page 98 The Lutheran Hymnal , hymn # 457 References ( edit ) Wikisource has original text
related to this article : What a Friend We Have in Jesus Jump up ^ `` What a Friend We Have in Jesus '' . Retrieved 2007 - 05 - 03 . Jump up ^ Polack , W.G. (
1941 ) . Handbook to the Lutheran Hymnal . St. Louis : Concordia . p. 323 . Jump up ^ `` Washington Phillips discography '' . wirz.de . Retrieved August 24 ,
2015 . Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic .
</item 6>

<item 7>
What a Friend We Have in Jesus
What a Friend We Have in Jesus - wikipedia What a Friend We Have in Jesus Jump to : navigation , search `` What a Friend We Have in Jesus '' Song Genre Hymn
Composer ( s ) Charles C. Converse ( 1868 ) Lyricist ( s ) Joseph M. Scriven ( 1855 ) `` What a Friend We Have in Jesus '' is a Christian hymn originally
written by Joseph M. Scriven as a poem in 1855 to comfort his mother who was living in Ireland while he was in Canada . Scriven originally published the poem
anonymously , and only received full credit for it in the 1880s . The tune to the hymn was composed by Charles Crozat Converse in 1868 . William Bolcom composed
a setting of the hymn . The hymn also has many versions with different lyrics in multiple languages . The Handbook to the Lutheran Hymnal notes , `` In spite of
the fact that this hymn , with its tune , has been criticized as being too much on the order of the sentimental gospel type , its popularity remains strong ,
and the hymn retains a place in modern hymnals . ''
</item 7>

<item 8>
What a Friend We Have in Jesus
It is translated as `` Preethi gala mana Yesu '' and it is listed in the Andhra Kraisthava Keerthanalu ( Andhra Christian hymnals ) as hymn No. 407 . In India ,
the hymn sung in Konkani , `` Dulob Jezu '' , was translated by Felix Paul Noronha of Mangalore . In popular culture ( edit ) This section indiscriminately
collects miscellaneous information . Please compress this material to remove any irrelevant or unimportant information . ( February 2018 ) A version was sung
during World War I with the line `` When this bloody war is over '' in place of the original title . The tune was used in the 1969 film , Oh ! What a Lovely War
with the title `` When This Lousy War Is Over '' . The tune was used in Alan Price 's song Changes , which was written for the 1973 film O Lucky Man ! and later
became a hit when used in the 1987 Volkswagen Golf television commercial . The tune was in the labor protest movement due to its wide popularity .
</item 8>

<item 9>
What a Friend We Have in Jesus
and piano score for `` What a Friend We Have in Jesus '' at HymnSite.com `` What a Friend We Have in Jesus '' Mp3 and lyrics ( hide ) Washington Phillips
Original Columbia 78 rpm records ( chronological ) `` Lift Him Up That 's All '' / `` Take Your Burden to the Lord and Leave It There '' `` Denomination Blues
-- Part 1 '' / `` Denomination Blues -- Part 2 '' `` Mother 's Last Word to Her Son '' / `` Paul and Silas in Jail '' `` What Are They Doing in Heaven Today ''
/ `` Jesus Is My Friend '' `` I Am Born to Preach the Gospel '' / `` Train Your Child '' `` A Mother 's Last Word to Her Daughter '' / `` I 've Got the Key to
the Kingdom '' `` I Had a Good Father and Mother '' / `` The Church Needs Good Deacons '' Later song releases `` You Ca n't Stop a Tattler -- Part 1 '' `` You
Ca n't Stop a Tattler -- Part 2 '' Related articles List of songs ( with recording & catalogue details ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
What a Friend We Have in Jesus
Glen Campbell ( 1989 , Favorite Hymns ) Driving Miss Daisy ( 1989 ) sung at Little Friendship Baptist Church in Atlanta , Georgia Ike & Tina Turner ( 1974 , The
Gospel According to Ike & Tina ) The John Tesh Project ( 2000 , Pure Hymns ) Amy Grant recorded it as part of the medley `` What a Friend We Have in Jesus / Old
Rugged Cross / How Great Thou Art '' on her 2002 studio album Legacy ... Hymns and Faith , and later included on her 2015 compilation album Be Still and Know
... Hymns & Faith . Alan Jackson ( 2006 , Precious Memories ) Brad Paisley ( 2008 , Play ) Ronnie Milsap ( 2009 , Then Sings My Soul ) Monty Alexander ( 2013 ,
Uplift 2 ) In asia ( edit ) In Japan , the hymn 's title was originally translated as `` Itsukushimi Fukaki '' ( いつくしみ 深き , lit . `` Merciful '' ) , which is
what it is best known by .
</item 10>
=== END ITEM id=687a59c4e36c2234 ===

=== ITEM id=685b46c14285686f ===
Gold sentences and facts:
Sentence 0: Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years elapsed 304 123 47 50
  Fact 0.0: The starting value is 500 million.
  Fact 0.1: The population values, in billions, are 0.5 and 8.
  Fact 0.2: The year values are 1500, 1804, 1927 and 2024.
  Fact 0.3: The years elapsed values are 304, 123, 47 and 50.

Retrieved items:
<item 1>
World population
Projected figures vary depending on underlying statistical assumptions and the variables used in projection calculations , especially the fertility variable .
Long - range predictions to 2150 range from a population decline to 3.2 billion in the `` low scenario '' , to `` high scenarios '' of 24.8 billion . One
extreme scenario predicted a massive increase to 256 billion by 2150 , assuming the global fertility rate remained at its 1995 level of 3.04 children per woman
; however , by 2010 the global fertility rate had declined to 2.52 . There is no estimation for the exact day or month the world 's population surpassed one or
two billion . The points at which it reached three and four billion were not officially noted , but the International Database of the United States Census
Bureau placed them in July 1959 and April 1974 respectively . The United Nations did determine , and commemorate , the `` Day of 5 Billion '' on July 11 , 1987
, and the `` Day of 6 Billion '' on October 12 , 1999 . The Population Division of the United Nations declared the `` Day of 7 Billion '' to be October 31 ,
2011 .
</item 1>

<item 2>
World population
The vertical axis is logarithmic and is in millions of people . Population size fluctuates at differing rates in differing regions . Nonetheless , population
growth is the long - standing trend on all inhabited continents , as well as in most individual states . During the 20th century , the global population saw its
greatest increase in known history , rising from about 1.6 billion in 1900 to over 6 billion in 2000 . A number of factors contributed to this increase ,
including the lessening of the mortality rate in many countries by improved sanitation and medical advances , and a massive increase in agricultural
productivity attributed to the Green Revolution . In 2000 , the United Nations estimated that the world 's population was growing at an annual rate of 1.14 % (
equivalent to around 75 million people ) , down from a peak of 88 million per year in 1989 . By 2000 , there were approximately ten times as many people on
Earth as there had been in 1700 .
</item 2>

<item 3>
World population
The transition from hyperbolic growth to slower rates of growth is related to the demographic transition . According to the Russian demographer Sergey Kapitsa ,
the world population grew between 67,000 BC and 1965 according to the following formula : N = C τ arccot ⁡ T 0 − T τ ( \ displaystyle N = ( \ frac ( C ) ( \ tau
) ) \ operatorname ( arccot ) ( \ frac ( T_ ( 0 ) - T ) ( \ tau ) ) ) where N is current population T is the current year C = ( 1.86 ± 0.01 ) 10 T = 2007 ± 1 τ
( \ displaystyle \ tau ) = 42 ± 1 Years for world population to double ( edit ) According to linear interpolation and extrapolation of UNDESA population
estimates , the world population has doubled , or will double , in the years listed in the tables below ( with two different starting points ) . During the 2nd
millennium , each doubling took roughly half as long as the previous doubling , fitting the hyperbolic growth model mentioned above .
</item 3>

<item 4>
World population
Milestones by the billions ( edit ) Main article : World population milestones World population milestones in billions Population 5 6 7 8 9 Year 1804 1927 1960
1999 2011 2024 2042 Years elapsed -- -- 123 33 14 13 12 12 13 18 It is estimated that the world population reached one billion for the first time in 1804 . It
was another 123 years before it reached two billion in 1927 , but it took only 33 years to reach three billion in 1960 . Thereafter , the global population
reached four billion in 1974 , five billion in 1987 , six billion in 1999 and , according to the United States Census Bureau , seven billion in March 2012 . The
United Nations , however , estimated that the world population reached seven billion in October 2011 . According to current projections , the global population
will reach eight billion by 2024 , and will likely reach around nine billion by 2042 . Alternative scenarios for 2050 range from a low of 7.4 billion to a high
of more than 10.6 billion .
</item 4>

<item 5>
World population
000 BC < 0.015 0 0 10,000 BC 8000 BC 5 6500 BC 5 5000 BC 5 4000 BC 7 3000 BC 14 2000 BC 27 1000 BC 50 7 33 9 500 BC 100 14 66 16 AD 1 200 23 141 28 1000 400 70
269 50 8 1500 458 86 243 84 39 1600 580 114 339 111 10 1700 682 106 436 125 10 1750 791 106 502 163 16 1800 1,000 107 656 203 24 7 1850 1,262 111 809 276 38 26
1900 1,650 133 947 408 74 82 6 1950 2,525 229 1,394 549 169 172 12.7 1955 2,758 254 1,534 577 193 187 14.
</item 5>

<item 6>
Graph query row:
entity: World population
relationship: RELATED_TO
milestone: World population reached one billion
date: 1804
</item 6>

<item 7>
World population
758 254 1,534 577 193 187 14.2 1960 3,018 285 1,687 606 221 204 15.8 1965 3,322 322 1,875 635 254 219 17.5 1970 3,682 366 2,120 657 288 231 19.7 4,061 416 2,378
677 326 242 21.5 1980 4,440 478 2,626 694 365 254 23.0 1985 4,853 550 2,897 708 406 267 24.9 5,310 632 3,202 721 447 281 27.0 1995 5,735 720 3,475 728 487 296
29.1 2000 6,127 814 3,714 726 527 314 31.1 2005 6,520 920 3,945 729 564 329 33.4 6,930 1,044 4,
</item 7>

<item 8>
World population
Archived from the original on May 11 , 2009 . Retrieved 2013 - 07 - 26 . Jump up ^ Lutz , Wolfgang ; Sanderson , Warren ; Scherbov , Sergei ( 1997 - 06 - 19 ) .
`` Doubling of world population unlikely '' . Nature . 387 ( 6635 ) : 803 -- 805 . doi : 10.1038 / 42935 . PMID 9194559 . Retrieved 2012 - 10 - 23 . ^ Jump up
to : `` World faces ' perfect storm ' of problems by 2030 , chief scientist to warn '' . The Guardian . March 18 , 2009 . Retrieved February 18 , 2013 . Jump up
^ Jowit , Juliette ( October 23 , 2011 ) . `` Paul Ehrlich , a prophet of global population doom who is gloomier than ever '' . The Guardian . London .
Retrieved July 22 , 2013 .
</item 8>

<item 9>
Graph query row:
entity: World population
relationship: RELATED_TO
milestone: World population reached two billion
date: 1927
</item 9>

<item 10>
World population
However , after 2024 , it is unlikely that there will be another doubling of the global population in the 21st century . Historic chart showing the periods of
time the world population has taken to double , from 1700 to 2000 Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years
elapsed 304 123 47 50 Starting at 375 million Population ( in billions ) 0.375 0.75 1.5 6 Year 1171 1715 1881 1960 1999 Years elapsed 544 166 79 39
Overpopulation ( edit ) Main articles : Human overpopulation and Malthusian catastrophe Predictions of scarcity ( edit ) Greater Los Angeles lies on a coastal
mediterranean savannah with a small watershed that is able to support at most one million people on its own water ; as of 2015 , the area has a population of
over 18 million . Researchers predict that similar cases of resource scarcity will grow more common as the world population increases .
</item 10>
=== END ITEM id=685b46c14285686f ===

=== ITEM id=7dedd22828357951 ===
Gold sentences and facts:
Sentence 0: `` You 'll Never Find Another Love Like Mine '' ( written by Kenny Gamble & Leon Huff ) is a song performed by R&B singer Lou Rawls on his 1976 album All Things in Time . The song proved to be Rawls ' breakthrough hit , reaching number one on both the R&B and Easy Listening charts as well as number four on the dance chart and number two on the US Billboard Hot 100 , where it was kept from the top spot for two weeks by `` You Should Be Dancing '' by The Bee Gees and `` ( Shake , Shake , Shake ) Shake Your Booty '' by KC and the Sunshine Band , respectively . This was the first and only time that one of Rawls ' records managed to reach Billboard 's pop top ten . It was the first big hit for Philadelphia International to feature the reformulated MFSB , after many of the original members left Gamble and Huff for better opportunities . The song started Rawls ' live shows from 1977 on .
  Fact 0.0: "You'll Never Find Another Love Like Mine" was written by Kenny Gamble and Leon Huff.
  Fact 0.1: "You'll Never Find Another Love Like Mine" is a song performed by R&B singer Lou Rawls.
  Fact 0.2: "You'll Never Find Another Love Like Mine" is on Lou Rawls' 1976 album All Things in Time.
  Fact 0.3: "You'll Never Find Another Love Like Mine" proved to be Rawls' breakthrough hit.
  Fact 0.4: "You'll Never Find Another Love Like Mine" reached number one on the R&B chart.
  Fact 0.5: "You'll Never Find Another Love Like Mine" reached number one on the Easy Listening chart.
  Fact 0.6: "You'll Never Find Another Love Like Mine" reached number four on the dance chart.
  Fact 0.7: "You'll Never Find Another Love Like Mine" reached number two on the US Billboard Hot 100.
  Fact 0.8: On the Billboard Hot 100, "You'll Never Find Another Love Like Mine" was kept from the top spot for two weeks by "You Should Be Dancing" by The Bee Gees and by "(Shake, Shake, Shake) Shake Your Booty" by KC and the Sunshine Band, respectively.
  Fact 0.9: "You'll Never Find Another Love Like Mine" was the first and only time one of Rawls' records managed to reach Billboard's pop top ten.
  Fact 0.10: "You'll Never Find Another Love Like Mine" was the first big hit for Philadelphia International to feature the reformulated MFSB.
  Fact 0.11: Many of the original MFSB members left Gamble and Huff for better opportunities.
  Fact 0.12: The song started Lou Rawls' live shows from 1977 on.

Retrieved items:
<item 1>
You'll Never Find Another Love Like Mine
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
You'll Never Find Another Love Like Mine
Contents ( hide ) 1 Chart performance 1.1 Weekly charts 1.2 Year - end charts 2 Later uses 3 Cover versions 4 See also 5 References 6 External links Chart
performance ( edit ) Weekly charts ( edit ) Chart ( 1976 -- 77 ) Peak position Australia KMR 22 Canada RPM Top Singles UK 10 US Billboard Hot 100 US Billboard
Hot Dance Club Play US Billboard Hot Adult Contemporary Tracks US Billboard Hot Black Singles Year - end charts ( edit ) Chart ( 1976 ) Rank Canada 38 US
Billboard Hot 100 32 Chart ( 1977 ) Rank Australia 97 Later uses ( edit ) It can also be heard in the films The Hot Chick ( 2002 ) , Guess Who ( 2005 ) ,
Disturbia ( 2007 ) , Ice Age : Dawn of the Dinosaurs ( 2009 ) , Hit and Run ( 2012 ) , Veronica Mars ( 2014 ) , and on the TV sitcoms My Wife and Kids ( 2001 --
2005 ) , in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy ,
</item 2>

<item 3>
You'll Never Find Another Love Like Mine
You 've Heard It All Let Me Be Good to You Sit Down and Talk to Me Singles `` Love Is a Hurtin ' Thing '' `` Your Good Thing ( Is About to End ) '' `` Bring It
On Home to Me '' `` You 'll Never Find Another Love Like Mine '' `` Lady Love '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=You%27ll_Never_Find_Another_Love_Like_Mine&oldid=803225603 '' Categories : 1976 singles 1976 songs Billboard Adult
Contemporary number - one singles Billboard Hot R&B / Hip - Hop Songs number - one singles Lou Rawls songs Philadelphia International Records singles Songs
written by Kenny Gamble Songs written by Leon Huff Songs from animated films Hidden categories : All articles with dead external links Articles with dead
external links from July 2016 Articles with permanently dead external links Articles with hAudio microformats Talk Contents About Wikipedia Italiano Português
Edit links This page was last edited on 1 October 2017 , at 07 : 04 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional
terms may apply .
</item 3>

<item 4>
Graph query row:
entity: Lou Rawls
relationship: RELATED_TO
role: None
relationship_detail: None
relation: None
fact: None
result: None
outcome: None
connected_entity: You'll Never Find Another Love Like Mine
</item 4>

<item 5>
You'll Never Find Another Love Like Mine
Cover versions ( edit ) `` You 'll Never Find Another Love Like Mine '' has been covered by singers Michael Bublé , Laura Pausini , reggae singer John Holt and
by the Dub Pistols ( who use a sample of John Holt 's version ) on their Speakers and Tweeters album . Jazz saxophonist Stanley Turrentine covered the song on
his 1976 album The Man with the Sad Face . as did Hank Crawford on Hank Crawford 's Back . This song was also featured on The Proud Family Soundtrack . See also
( edit ) List of number - one R&B singles of 1976 ( U.S. ) List of number - one adult contemporary singles of 1976 ( U.S. ) References ( edit ) Jump up ^
Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 483 . Jump up ^ Whitburn , Joel ( 2004 ) . Hot Dance / Disco : 1974
- 2003 . Record Research . p. 213 .
</item 5>

<item 6>
You'll Never Find Another Love Like Mine
The song proved to be Rawls ' breakthrough hit , reaching number one on both the R&B and Easy Listening charts as well as number four on the dance chart and
number two on the US Billboard Hot 100 , where it was kept from the top spot for two weeks by `` You Should Be Dancing '' by The Bee Gees and `` ( Shake , Shake
, Shake ) Shake Your Booty '' by KC and the Sunshine Band , respectively . This was the first and only time that one of Rawls ' records managed to reach
Billboard 's pop top ten . It was the first big hit for Philadelphia International to feature the reformulated MFSB , after many of the original members left
Gamble and Huff for better opportunities . The song started Rawls ' live shows from 1977 on . The single went on to sell over a million copies and was certified
gold by the RIAA . Contents ( hide ) 1 Chart performance 1.1 Weekly charts 1.
</item 6>

<item 7>
You'll Never Find Another Love Like Mine
in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy , That ' 70s Show ( 1998 -- 2006 ) , South Park ( 1997 -- ) , Two and a Half Men ( 2003
-- 2015 ) , and Psych ( 2006 -- 2014 ) . The song could also be heard in an episode of The Proud Family , where Lou Rawls sings and dances with Penny Proud . In
addition , it was performed by the singer in doll form in an Action League Now ! segment on KaBlam ! . The song `` If You Could Love Me '' by Edwyn Collins uses
many of the same chords , and has a similar sound . Rocco DiSpirito and Karina Smirnoff also danced the rumba to this song in week 2 of season 7 of Dancing with
the Stars .
</item 7>

<item 8>
You'll Never Find Another Love Like Mine
Top Adult Contemporary : 1961 - 2001 . Record Research . p. 200 . Jump up ^ `` Top Singles -- Volume 26 , No. 14 & 15 , January 08 1977 '' . RPM . Library and
Archives Canada . Retrieved June 13 , 2016 . Jump up ^ `` Top 100 Hits of 1976 / Top 100 Songs of 1976 '' . Musicoutfitters.com . Retrieved 2016 - 10 - 12 .
Jump up ^ Erlewine , Michael . The Man with the Sad Face -- Review at AllMusic . Retrieved April 16 , 2015 . External links ( edit ) Lyrics of this song at
MetroLyrics ( hide ) Lou Rawls Albums Stormy Monday All Things in Time Unmistakably Lou When You Hear Lou ,
</item 8>

<item 9>
You'll Never Find Another Love Like Mine
You 'll Never Find Another Love Like Mine - wikipedia You 'll Never Find Another Love Like Mine Jump to : navigation , search `` You 'll Never Find Another Love
Like Mine '' Single by Lou Rawls from the album All Things in Time B - side `` Let 's Fall in Love All Over Again '' Released May 1976 Format 7 '' Genre R&B ,
jazz , disco Length 4 : 26 ( album version ) 3 : 36 ( single edit ) Label Philadelphia International Records Songwriter ( s ) Kenny Gamble Leon Huff Producer (
s ) Kenny Gamble Leon Huff Lou Rawls singles chronology `` A Natural Man '' ( 1971 ) `` You 'll Never Find Another Love Like Mine '' ( 1976 ) `` Groovy People
'' ( 1976 ) `` A Natural Man '' ( 1971 ) `` You 'll Never Find Another Love Like Mine '' ( 1976 ) `` Groovy People '' ( 1976 ) `` You 'll Never Find Another
Love Like Mine '' ( written by Kenny Gamble & Leon Huff ) is a song performed by R&B singer Lou Rawls on his 1976 album All Things in Time .
</item 9>

<item 10>
You'll Never Find Another Love Like Mine
Record Research . p. 213 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 04 . Retrieved 2015 - 06 - 07 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 11 .
Retrieved 2015 - 06 - 07 . Jump up ^ `` The Hits Just Keep On Comin ' : You 'll Never Find Another Love Like Mine '' . Hitsjustkeeponcomin.blogspot.com. 2005 -
01 - 12 . Retrieved 2016 - 10 - 12 . Jump up ^ ( 1 ) Jump up ^ `` Archived copy '' . Archived from the original on June 2 , 2016 . Retrieved October 27 , 2015 .
Jump up ^ `` Image : RPM Weekly - Library and Archives Canada '' . Bac-lac.gc.ca . Retrieved 2016 - 10 - 12 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult
Contemporary : 1961 - 2001 . Record Research .
</item 10>
=== END ITEM id=7dedd22828357951 ===
