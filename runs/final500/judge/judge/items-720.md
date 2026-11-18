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

=== ITEM id=43ce1f31f0bf367b ===
Gold sentences and facts:
Sentence 0: `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period .
  Fact 0.0: Fore is originally an Australian interjection.
  Fact 0.1: Fore is used to warn anyone standing or moving in the flight of a golf ball.
  Fact 0.2: An 1881 Australian Golf Museum mention of the term exists.
  Fact 0.3: The 1881 mention indicates that the term Fore was in use at least as early as 1881.

Retrieved items:
<item 1>
God Save the Queen
The phrase `` God Save the King '' is much older than the song , appearing , for instance , several times in the King James Bible . A text based on the 1st Book
of Kings Chapter 1 : verses 38 -- 40 , `` ... And all the people rejoic 'd , and said : God save the King ! Long live the King ! May the King live for ever ,
Amen '' , has been sung at every coronation since that of King Edgar in 973 . Scholes says that as early as 1545 `` God Save the King '' was a watchword of the
Royal Navy , with the response being `` Long to reign over us '' . He also notes that the prayer read in churches on anniversaries of the Gunpowder Plot
includes words which might have formed part of the basis for the second verse `` Scatter our enemies ... assuage their malice and confound their devices '' . In
1745 , The Gentleman 's Magazine published `` God save our lord the king : A new song set for two voices '' , describing it `` As sung at both Playhouses '' (
the Theatres Royal at Drury Lane and Covent Garden ) .
</item 1>

<item 2>
Fore (golf)
Fore ( golf ) - Wikipedia Fore ( golf ) Jump to : navigation , search The tough rubber core of a golf ball makes it a hazard to others following a wayward shot
, despite its weight not exceeding 45.93 g ( 1.62 oz ) `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the
flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period . It is
believed to come from the military `` beware before '' , which an artilleryman about to fire would yell alerting nearby infantrymen to drop to the ground to
avoid the shells overhead . ( Before may mean `` in front of ( the gun being fired ) '' ; fore may mean `` ( look ) ahead '' . ) Other possible origins include
the term being derived from the term `` fore - caddie '' , a caddie waiting down range from the golfer to find where the ball lands .
</item 2>

<item 3>
If a tree falls in a forest
If a tree falls in a forest - wikipedia If a tree falls in a forest Jump to : navigation , search For the 2011 documentary , see If a Tree Falls : A Story of
the Earth Liberation Front . A fallen tree in a forest `` If a tree falls in a forest and no one is around to hear it , does it make a sound ? '' is a
philosophical thought experiment that raises questions regarding observation and perception . Contents ( hide ) 1 History 2 Metaphysics 2.1 The possibility of
unperceived existence 2.2 Knowledge of the unobserved world 2.3 The dissimilarity between sensation and reality 2.4 Buddhist perspectives 2.4. 1 Tibetan 2.4. 2
Zen 2.4. 2.1 Hui - neng 's Flag 2.4.
</item 3>

<item 4>
Orc (Middle-earth)
Christopher Tolkien notes that while the author clearly differentiates between `` goblins and Orcs '' in the Tale of Tinúviel , the two terms appear to be
synonymous in the Tale of Turambar . The word Gongs is also used on a few occasions ; it appears both distinct from Orcs and as a sub-type of Orc . Christopher
Tolkien remarks that Gongs are `` evil beings obscurely related to Orcs '' . Both goblins and Orcs are occasionally mentioned as being `` of Melkor '' and also
acting independently . Orcs , goblins , and gongs appear in Tolkien 's two lexicons of elvish languages . The Qenya Lexicon from approximately 1915 defines Orc
as meaning `` monster , demon '' , and the Gnomish Lexicon dated 1917 defines Orc as `` goblin '' and Gong as `` one of a tribe of the Orcs , a goblin '' .
Christopher Tolkien also notes that in the latter lexicon , the word Gnome ( later Noldor ) is an emendation from Goblin .
</item 4>

<item 5>
Orc (Middle-earth)
They were armed with short broad - bladed swords , not with the curved scimitars usual with Orcs : and they had bows of yew , in length and shape like the bows
of Men . And : Upon a stake in the middle was set a great goblin head ; upon its shattered helm the white badge could still be seen . The `` white badge ''
mentioned in the latter passage makes it clear that the beheaded goblin was one of Saruman 's Uruk - hai . Tolkien writes that these bore a white Elf - rune
with the value of `` S '' on their helmets . Tolkien also wrote the following note , appearing in some editions of The Hobbit : Orc is not an English word . It
occurs in one or two places but is usually translated goblin ( or hobgoblin for the larger kinds ) . Orc is the hobbits ' form of the name given at that time to
these creatures , and it is not connected at all with orc , ork , applied to sea - animals of dolphin - kind .
</item 5>

<item 6>
Fore (golf)
These caddies were often warned about oncoming golf balls by a shout of the term `` fore - caddie '' which was eventually shortened to just `` fore ! '' . The
Colonel Bogey March is based on the descending minor third which the original Colonel Bogey whistled instead of yelling Fore around 1914 . Notes ( edit ) Jump
up ^ SND : Fore ^ Jump up to : Why Do Golfers Yell `` Fore '' for Errant Shots ? retrieved June 15 , 2007 Jump up ^ Derivation 4 Fore ! and Caddie retrieved
June 15 , 2007 Jump up ^ The Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 33 , no . 192 ( December 1910 ) References ( edit ) Scott , James
Sibbald David , The British Army : Its Origin , Progress , and Equipment , 1868 Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 32 , no . 292 (
December 1910 ) This golf - related article is a stub . You can help Wikipedia by expanding it .
</item 6>

<item 7>
Fore (golf)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Fore_(golf)&oldid=802796028 '' Categories : Golf
terminology Scottish English Golf stubs Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked
weasel - worded phrases from January 2016 All stub articles Talk About Wikipedia Hrvatski Italiano Norsk Srpskohrvatski / српскохрватски Svenska Edit links This
page was last edited on 28 September 2017 , at 14 : 56 . About Wikipedia
</item 7>

<item 8>
Pencil
Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people
have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the
element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally
mean lead pen . The value of graphite would soon be realised to be enormous , mainly because it could be used to line the moulds for cannonballs ; the mines
were taken over by the Crown and were guarded . When sufficient stores of graphite had been accumulated , the mines were flooded to prevent theft until more was
required . The usefulness of graphite for pencils was discovered as well , but graphite for pencils had to be smuggled . The news of the usefulness of these
early pencils spread far and wide , attracting the attention of artists all over the known world . Because graphite is soft , it requires some form of
encasement .
</item 8>

<item 9>
Orc (Middle-earth)
About the goblins of The Hobbit , Tolkien wrote : They are not based on direct experience of mine ; but owe , I suppose , a good deal to the goblin tradition
... especially as it appears in George MacDonald , except for the soft feet which I never believed in . Middle - earth terms for Orcs ( edit ) In the High -
elven tongue Quenya , the word for `` Orc '' was urco , plural urqui , meaning `` bogey '' , or `` bogeyman '' , that is , something that provokes fear . In the
Grey - elven tongue Sindarin , it was orch , plural yrch . In the Dwarven tongue Khuzdul , it was rukhs , plural rakhâs . In the language of the Drúedain or
Wild Men , it was gorgûn . In the Black Speech of Mordor , the equivalent was Uruk , as can be seen in Uruk - hai , `` Orc - folk '' . Orc itself is from
Rohirric and the Hobbit - language , which shared linguistic roots , but the term is clearly related to the older Elvish words .
</item 9>

<item 10>
Forensic firearm examination
Class characteristics ( edit ) Preliminary examination of the bullet can exclude a large number of weapons by examining the general characteristics of a
recovered bullet . By determining general aspects of the fired ammunition , a number of weapons can be immediately excluded as being incapable of firing that
type of bullet . The make and model of the weapon can also be inferred from the combination of different class characteristics that are common to specific
manufactures . The three main class characteristics of all bullets are the lands and grooves , the caliber of the bullet , and the rifling twist . All three can
be tied directly to the type of barrel that was used to fire the bullet . The lands and grooves of barrel are the bumps and valleys created when the rifling is
created . The caliber is the diameter of the barrel . The twist is the direction of the striations left by the barrel 's rifling , clockwise ( right - handed )
or counterclockwise ( left - handed ) . Most barrels will have a right - handed twist with the exception of weapons created by the Colt 's Manufacturing Company
which uses left - handed twists .
</item 10>
=== END ITEM id=43ce1f31f0bf367b ===

=== ITEM id=9b4af448611e12bc ===
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
Volcanic Productions . p. 389 . ISBN 0 - 646 - 44865 - X . Jump up ^ Cavanough , Maurice , `` The Melbourne Cup '' , Jack Pollard P / L , North Sydney , 1976
Jump up ^ `` The Most Famous Racehorses in History '' . Retrieved 4 February 2011 . ^ Jump up to : Flemington Archived 14 July 2011 at the Wayback Machine .
Jump up ^ The Masters Touch , Racing with Etienne de Mestre , Winner of 5 Melbourne Cups by Keith W. Paterson , Published by Keith W. Paterson ( Nowra ) ( 2008
) ( ISBN 9780646500287 ) , Page 46 Jump up ^ Race May 13th 2008 09 : 34 ( 2008 - 05 - 13 ) . `` Aboriginal jockeys who won Melbourne Cup '' . Racingwrite.com .
Retrieved 2014 - 02 - 01 . Jump up ^ `` The wonder filly and a boy who carved their names into history '' . The Sydney Morning Herald . 1 November 2004 .
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
1 Melbourne Cup Tour 3 History 3.1 Early years 3.2 Recent years 3.3 Timeline of notable events 3.4 Public holiday 4 Results and records 4.1 Most wins by a horse
4.2 Most wins by a jockey 4.3 Most wins by a trainer 4.4 Most wins by an owner 4.5 Other records 5 Attendance 6 Off the track 7 See also 8 Notes 9 Further
reading 10 External links Qualifying and race conditions ( edit ) 1932 & 1934 winner Peter Pan ridden by jockey Darby Munro , won twice The race is a quality
handicap for horses 3 years old and over , run over a distance of 3,200 metres , on the first Tuesday in November at Flemington Racecourse . The minimum
handicap weight is 50 kg . There is no maximum weight , but the top allocated weight must not be less than 57 kg . The weight allocated to each horse is
declared by the VRC Handicapper in early September .
</item 4>

<item 5>
Melbourne Cup
Subsequent foreign bred horses to win Cup were Backwood 1924 ; Phar Lap 1930 ; Wotan 1936 Belldale Ball 1980 ; At Talaq 1986 ; Kingston Rule 1990 ; Vintage Crop
1993 ; Jeune 1994 ; Media Puzzle 2002 ; Makybe Diva 2003 , 2004 , 2005 ; Americain 2010 and Dunaden 2011 . The 1938 Melbourne Cup was won by trainer Mrs. Allan
McDonald , who conditioned Catalogue . Mrs McDonald was a successful trainer in New Zealand , however , at the time women were not allowed to compete as
trainers in Australia so her husband 's name was officially recorded as the winning trainer . The 2001 edition was won by New Zealand mare Ethereal , trained by
Sheila Laxon , the first woman to formally train a Melbourne Cup winner . She also won the Caulfield Cup , a 2,400 metre race also held in Melbourne , and
therefore has won the `` Cups Double '' .
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
The race was won by the George Hanlon trained 40 - 1 outsider Piping Lane ridden by John Letts and carrying only 48 kg . 1985 -- The first sponsored Melbourne
Cup , and the first million dollar Cup , with $650,000 for the winner . 1987 -- First female jockey to ride in the cup was Maree Lyndon on Argonaut Style .
Dunaden : 2011 Melbourne Cup winner , painted by Charles Church 1993 -- The Dermot Weld trained Irish gelding Vintage Crop became the first Northern hemisphere
trained horse to win the Melbourne Cup and the first Irish horse . 2001 -- Sheila Laxon , was the first woman trainer to officially win the Melbourne Cup .
However , Mrs. A. McDonald ( 1938 ) with Catalogue was really the first woman trainer to win . Women then could not be registered as trainers in Australia , and
it was her husband who was the registered trainer . Mrs. McDonald 's win was as a female trainer of a female owned horse . 2003 -- First Australian female
jockey to ride in the cup was Clare Lindop on Debben .
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
1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman (
1989 , 1992 , 1995 , 2004 , 2005 ) Most wins by an owner ( edit ) 6 -- Lloyd Williams ( 1981 , 1985 , 2007 , 2012 , 2016 , 2017 ) 4 -- John Tait ( 1866 , 1868 ,
1871 , 1872 ) 4 -- Etienne de Mestre ( 1861 * , 1862 * , 1867 * , 1878 ) * The winners of these years were leased . 4 -- Dato Tan Chin Nam ( 1974 , 1975 , 1996
, 2008 ) Other records ( edit ) Record winning time -- 3.16.
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
=== END ITEM id=9b4af448611e12bc ===

=== ITEM id=4fcf56df48233b3b ===
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
He also hand - picked the recording band for the song . `` Total Eclipse of the Heart '' had to be shortened for radio play . Tyler did not believe that the
song was radio - friendly at its full length ; the song was reduced from seven minutes and two seconds to four minutes and thirty seconds . The power ballad
remains Tyler 's most successful song , peaking at No. 1 in the United States , South Africa , Australia , New Zealand , Canada and the United Kingdom . At its
peak , it sold 60,000 copies per day , and approximately 6 million copies in total . It won the Variety Club award in the UK for best single of 1983 . The song
also made number 82 of VH1 's top 100 love songs . `` Total Eclipse of the Heart '' A sample of `` Total Eclipse of the Heart '' , performed by Bonnie Tyler .
Problems playing this file ? See media help . Tyler told Record Mirror that she thought the song was about `` someone who wants to love so badly she 's lying
there in complete darkness . ''
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
Total Eclipse of the Heart - wikipedia Total Eclipse of the Heart Jump to : navigation , search For the episode of `` The Vampire Diaries '' , see Total Eclipse
of the Heart ( The Vampire Diaries ) . `` Total Eclipse of the Heart '' Artwork for UK and some European releases Single by Bonnie Tyler from the album Faster
Than the Speed of Night B - side `` Take Me Back '' ( UK ) `` Straight from the Heart '' ( US ) Released Format 7 '' 12 '' CD Length 6 : 59 ( album version ) 5
: 53 ( 2013 re-recording for the album Rocks and Honey ) 5 : 32 ( music video version ) 4 : 35 ( 2011 single version ) 4 : 30 ( single version ) 3 : 49 ( 2005
re-recording for the album Wings ) Label Columbia Songwriter ( s ) Jim Steinman Producer ( s ) Jim Steinman Bonnie Tyler singles chronology `` Sayonara Tokyo ''
( 1981 ) `` Total Eclipse of the Heart '' ( 1983 ) `` Take Me Back '' ( 1983 ) `` Sayonara Tokyo '' ( 1981 ) `` Total Eclipse of the Heart
</item 10>
=== END ITEM id=4fcf56df48233b3b ===

=== ITEM id=d9bb9adc0d3b8975 ===
Gold sentences and facts:
Sentence 0: Somatic motor neurons originate in the central nervous system , project their axons to skeletal muscles ( such as the muscles of the limbs , abdominal , and intercostal muscles ) , which are involved in locomotion . The three types of these neurons are the alpha efferent neurons , beta efferent neurons , and gamma efferent neurons . They are called efferent to indicate the flow of information from the central nervous system ( CNS ) to the periphery .
  Fact 0.0: Somatic motor neurons originate in the central nervous system.
  Fact 0.1: Somatic motor neurons project their axons to skeletal muscles.
  Fact 0.2: Examples of those skeletal muscles are the muscles of the limbs, abdominal muscles, and intercostal muscles.
  Fact 0.3: The skeletal muscles that somatic motor neurons project to are involved in locomotion.
  Fact 0.4: There are three types of somatic motor neurons.
  Fact 0.5: The three types are alpha efferent neurons, beta efferent neurons, and gamma efferent neurons.
  Fact 0.6: Somatic motor neurons are called efferent.
  Fact 0.7: The name efferent indicates the flow of information from the central nervous system (CNS) to the periphery.
Sentence 1: The term ' motor neuron ' is usually restricted to the lower motor neurons , the efferent nerves that directly innervate muscles .
  Fact 1.0: The term 'motor neuron' is usually restricted to the lower motor neurons.
  Fact 1.1: Lower motor neurons are the efferent nerves that directly innervate muscles.

Retrieved items:
<item 1>
Motor neuron
the command of visceral muscles is disynaptic ( involving two neurons : the general visceral motor neuron located in the CNS , which synapses onto a ganglionic
neuron , located in the PNS , which synapses onto the muscle ) . All vertebrate motor neurons are cholinergic , that is , they release the neurotransmitter
acetylcholine . Parasympathetic ganglionic neurons are also cholinergic , whereas most sympathetic ganglionic neurons are noradrenergic , that is , they release
the neurotransmitter noradrenaline . ( see Table ) Motor neurons in Regenerative medicine ( edit ) Human lower motor neurons can be generated in vitro from
embryonic stem cells and induced pluripotent stem cells . See also ( edit ) Betz cell Central chromatolysis Motor dysfunction Nerve References ( edit ) Jump up
^ Tortora , Gerard ; Derrickson , Bryan ( 2014 ) . Principles of Anatomy & Physiology ( 14th ed . ) . New Jersey : John Wiley & Sons , Inc . pp. 406 , 502 , 541
. ISBN 978 - 1 - 118 - 34500 - 9 .
</item 1>

<item 2>
Motor neuron
There are three primary categories lower motor neurons , which can be further divided in sub-categories . According to their targets , motor neurons are
classified into three broad categories : Somatic motor neurons Special visceral motor neurons General visceral motor neurons Somatic motor neurons ( edit )
Somatic motor neurons originate in the central nervous system , project their axons to skeletal muscles ( such as the muscles of the limbs , abdominal , and
intercostal muscles ) , which are involved in locomotion . The three types of these neurons are the alpha efferent neurons , beta efferent neurons , and gamma
efferent neurons . They are called efferent to indicate the flow of information from the central nervous system ( CNS ) to the periphery . Alpha motor neurons
innervate extrafusal muscle fibers , which are the main force - generating component of a muscle . Their cell bodies are in the ventral horn of the spinal cord
and they are sometimes called ventral horn cells . A single motor neuron may synapse with 150 muscle fibers on average . The motor neuron and all of the muscle
fibers to which it connects is a motor unit .
</item 2>

<item 3>
Motor neuron
New Jersey : John Wiley & Sons , Inc . Jump up ^ Fitzpatrick , D. ( 2001 ) Lower Motor Neuron Circuits and Motor Control : Overview . In D. Purves , G.J.
Augustine , D. Fitzpatrick , et al. ( Ed . ) , Neuroscience . Retrieved from https://www.ncbi.nlm.nih.gov/books/NBK10979/ ^ Jump up to : `` CHAPTER NINE '' .
www.unc.edu . Retrieved 2017 - 12 - 08 . Jump up ^ Silverthorn , Dee Unglaub ( 2010 ) . Human Physiology : An Integrated Approach . Pearson . p. 398 . ISBN 978
- 0 - 321 - 55980 - 7 . ^ Jump up to : Tortora , G.J. , Derrickson , B. ( 2011 ) . Muscular Tissue . In B. Roesch , L. Elfers , K. Trost , et al. ( Ed . ) ,
Principles of Anatomy and Physiology ( pp. 305 - 307 , 311 ) .
</item 3>

<item 4>
Graph query row:
entity: Somatic motor neurons
relationship: RELATED_TO
related: Skeletal muscles
detail: projects_axons_to
</item 4>

<item 5>
Motor neuron
A tetanic contraction is caused by constant , very high frequency stimulation - the action potentials come at such a rapid rate that individual twitches are
indistinguishable , and tension rises smoothly eventually reaching a plateau . Contents ( hide ) 1 Development 2 Anatomy and physiology 2.1 Upper motor neurons
2.2 White Matter Tract 2.3 Lower Motor Neurons 2.4 Somatic motor neurons 2.4. 1 Neuromuscular Junction 2.5 Special visceral motor neurons 2.6 General visceral
motor neurons 3 Motor neurons in regenerative medicine 4 See also 5 References 6 Sources Development ( edit ) Motor neurons begin to develop early in embryonic
development , and motor function continues to develop well into childhood . In the neural tube cells are specified to either the rostral - caudal axis or
ventral - dorsal axis . The axons of motor neurons begin to appear in the fourth week of development from the ventral region of the ventral - dorsal axis ( the
basal plate ) . This homeodomain is known as the motor neural progenitor domain ( pMN ) .
</item 5>

<item 6>
Motor neuron
Muscle relaxation and inhibition of muscle contraction in vertebrates is obtained only by inhibition of the motor neuron itself . Muscle innervation may
eventually play a role in the maturation of motor activity . This is how muscle relaxants work by acting on the motor neurons that innervate muscles ( by
decreasing their electrophysiological activity ) or on cholinergic neuromuscular junctions , rather than on the muscles themselves . Special visceral motor
neurons ( edit ) These are also known as branchial motor neurons , which are involved in facial expression , mastication , phonation , and swallowing .
Associated cranial nerves are the oculomotor , abducens , trochlear , and hypoglossal nerves .
</item 6>

<item 7>
Motor neuron
Branch of NS Position Neurotransmitter Somatic n / a Acetylcholine Parasympathetic Preganglionic Acetylcholine Parasympathetic Ganglionic Acetylcholine
Sympathetic Preganglionic Acetylcholine Sympathetic Ganglionic Norepinephrine * * Except fibers to sweat glands and certain blood vessels Motor neuron
neurotransmitters General visceral motor neurons ( edit ) These motor neurons indirectly innervate cardiac muscle and smooth muscles of the viscera ( the
muscles of the arteries ) : they synapse onto neurons located in ganglia of the autonomic nervous system ( sympathetic and parasympathetic ) , located in the
peripheral nervous system ( PNS ) , which themselves directly innervate visceral muscles ( and also some gland cells ) . In consequence : the motor command of
skeletal and branchial muscles is monosynaptic ( involving only one motor neuron , respectively , somatic and branchial , which synapses onto the muscle ) .
</item 7>

<item 8>
Motor neuron
There are two types of motor neuron -- upper motor neurons and lower motor neurons . Axons from upper motor neurons synapse onto interneurons in the spinal cord
and occasionally directly onto lower motor neurons . The axons from the lower motor neurons are efferent nerve fibers that carry signals from the spinal cord to
the effectors . Types of lower motor neurons are alpha motor neurons , beta motor neurons , and gamma motor neurons . The term ' motor neuron ' is usually
restricted to the lower motor neurons , the efferent nerves that directly innervate muscles . A single motor neuron may innervate many muscle fibres and a
muscle fibre can undergo many action potentials in the time taken for a single muscle twitch . As a result , if an action potential arrives before a twitch has
completed , the twitches can superimpose on one another , either through summation or a tetanic contraction . In summation , the muscle is stimulated
repetitively such that additional action potentials coming from the somatic nervous system arrive before the end of the twitch . The twitches thus superimpose
on one another , leading to a force greater than that of a single twitch .
</item 8>

<item 9>
Motor neuron
Motor neuron - wikipedia Motor neuron Jump to : navigation , search Motor neuron Micrograph of the hypoglossal nucleus showing motor neurons with their
characteristic coarse Nissl substance ( `` tigroid '' cytoplasm ) . H&E - LFB stain . Details Location Ventral horn of the spinal cord , some cranial nerve
nuclei Shape Projection neuron Function Excitatory projection ( to NMJ ) Neurotransmitter UMN to LMN : glutamate ; LMN to NMJ : ACh Presynaptic connections
Primary motor cortex via the Corticospinal tract Postsynaptic connections Muscle fibers and other neurons Identifiers MeSH D009046 NeuroLex ID nifext_103 TA
A14. 2.00. 021 FMA 5867 Anatomical terms of neuroanatomy ( edit on Wikidata ) A motor neuron ( or motoneuron ) is a neuron whose cell body is located in the
motor cortex , brainstem or the spinal cord , and whose axon ( fiber ) projects to the spinal cord or outside of the spinal cord to directly or indirectly
control effector organs , mainly muscles and glands . There are two types of motor neuron -- upper motor neurons and lower motor neurons .
</item 9>

<item 10>
Motor neuron
These use both oxidative and glycolytic means to gain energy . In addition to voluntary skeletal muscle contraction , alpha motor neurons also contribute to
muscle tone , the continuous force generated by noncontracting muscle to oppose stretching . When a muscle is stretched , sensory neurons within the muscle
spindle detect the degree of stretch and send a signal to the CNS . The CNS activates alpha motor neurons in the spinal cord , which cause extrafusal muscle
fibers to contract and thereby resist further stretching . This process is also called the stretch reflex . Beta motor neurons innervate intrafusal muscle
fibers of muscle spindles , with collaterals to extrafusal fibres . There are two types of beta motor neurons : Slow Contracting - These innervate extrafusal
fibers . Fast Contracting - These innervate intrafusal fibers . Gamma motor neurons innervate intrafusal muscle fibers found within the muscle spindle . They
regulate the sensitivity of the spindle to muscle stretching . With activation of gamma neurons , intrafusal muscle fibers contract so that only a small stretch
is required to activate spindle sensory neurons and the stretch reflex .
</item 10>
=== END ITEM id=d9bb9adc0d3b8975 ===

=== ITEM id=fc34d9261e62485f ===
Gold sentences and facts:
Sentence 0: Year Film Role Notes Dumb and Dumber State Trooper Down Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr. Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P. Fudge My Life in Ruins Big Al Sawchuck 2009 The Haunted World of El Superbeasto Gerard the Exterminator ( voice ) 2011 Spooky Buddies Warwick the Warlock Lloyd the Conqueror Vulcan Dahmer Vs. Gacy God Dan Vs . Hiram Episode : Technology Robot Chicken Professor Cornelius Q. Quibblefingers ( voice ) 1 episode ; also writes 4 episodes 2012 The High Fructose Adventures of Annoying Orange Apple ( voice ) Robot and Monster Monster ( voice ) Main Role 2014 Back in the Day Skunk 2015 Jake and the Never Land Pirates Captain Frost ( voice ) Penn Zero : Part - Time Hero Helper Hue ( voice ) 2016 Sausage Party Ketchup Bottle / Drug Dealer / Baba Ganoush ( voice ) Skylanders Academy Hugo ( voice ) 2017 - present Puppy Dog Pals Bob ( voice ) Also show creator
  Fact 0.0: The listed performer had the role State Trooper in Dumb and Dumber.
  Fact 0.1: The listed performer appeared in Down Periscope, with no role given.
  Fact 0.2: The listed performer had the role 'Sonar' Lovacelli in E.T.
  Fact 0.3: The listed performer had the role Fred Z. Randall in RocketMan.
  Fact 0.4: The listed performer had the role Pet Wrangler in Wag the Dog.
  Fact 0.5: The listed performer had the role Himself in Space Ghost Coast to Coast, for 1 episode.
  Fact 0.6: In 1998, the listed performer had the role Kenny Davis in Half Baked.
  Fact 0.7: In 1998, the listed performer had the role Hitchhiker, uncredited, in There's Something About Mary.
  Fact 0.8: In 1998, the listed performer had the role Callum in Dog Park.
  Fact 0.9: In 1998, the listed performer had the role Newton, a voice role, in Ned's Newt.
  Fact 0.10: In 1998, the listed performer had the role Tucker in Mr. Headmistress.
  Fact 0.11: In 1999, the listed performer had the role Slater in Superstar.
  Fact 0.12: In 2000, the listed performer had the role Special Agent Steve Hanson in The Whole Nine Yards.
  Fact 0.13: In 2000, the listed performer had the role Officer Harry Cox in Big Money Hustlas.
  Fact 0.14: In 2000, the listed performer had the role Richard Breggs in Becoming Dick.
  Fact 0.15: In 2000, the listed performer had the role Darren in Freddy Got Fingered.
  Fact 0.16: In 2000, the listed performer had the role Mike Bonner, a voice role, in Gary & Mike.
  Fact 0.17: In 2002, the listed performer had the role Sheriff Hubbard in Back by Midnight.
  Fact 0.18: In 2002, the listed performer had the role Doofer / Roberta in Sorority Boys.
  Fact 0.19: In 2003, the listed performer had the role Jake in Family Tree.
  Fact 0.20: In 2003, the listed performer had the role Zee in Kart Racer.
  Fact 0.21: In 2005, the listed performer had the role Bleckman in Lucky 13.
  Fact 0.22: In 2005, the listed performer had the role Policeman in Because of Winn-Dixie.
  Fact 0.23: In 2005, the listed performer had the role Lugnut, a voice role, in Robots.
  Fact 0.24: In 2006, the listed performer had the role Russell Porpis-Gunders in Employee of the Month.
  Fact 0.25: In 2006, the listed performer had the role Rip in Surf School.
  Fact 0.26: In 2006, the listed performer had the role Head Dog Cop, a voice role, for 1 episode, in Ergo Proxy.
  Fact 0.27: In 2007, the listed performer had the role Carl, a voice role, in Meet the Robinsons.
  Fact 0.28: In 2007, the listed performer had the role Sheriff Rosco P. Coltrane in The Dukes of Hazzard: The Beginning.
  Fact 0.29: In 2008, the listed performer had the role Derek in Bachelor Party 2: The Last Temptation.
  Fact 0.30: In 2008, the listed performer had the role Giraffe, a voice role, in Madagascar: Escape 2 Africa.
  Fact 0.31: In 2009, the listed performer had the role Fudge Face Elmore P. Fudge in Fudgy Wudgy.
  Fact 0.32: In 2009, the listed performer had the role Big Al Sawchuck in My Life in Ruins.
  Fact 0.33: In 2009, the listed performer had the role Gerard the Exterminator, a voice role, in The Haunted World of El Superbeasto.
  Fact 0.34: In 2011, the listed performer had the role Warwick the Warlock, Lloyd the Conqueror, and Vulcan in Spooky Buddies.
  Fact 0.35: In 2011, the listed performer had the role God in Dahmer Vs. Gacy.
  Fact 0.36: In 2011, the listed performer had the role Robot in Dan Vs. Hiram, Episode: Technology.
  Fact 0.37: In 2011, the listed performer had the role Professor Cornelius Q. Quibblefingers, a voice role, for 1 episode, in Robot Chicken.
  Fact 0.38: The listed performer also wrote 4 episodes of Robot Chicken.
  Fact 0.39: In 2012, the listed performer had the role Apple, a voice role, in The High Fructose Adventures of Annoying Orange.
  Fact 0.40: In 2012, the listed performer had the role Monster, a voice role and a main role, in Robot and Monster.
  Fact 0.41: In 2014, the listed performer had the role Skunk in Back in the Day.
  Fact 0.42: In 2015, the listed performer had the role Captain Frost, a voice role, in Jake and the Never Land Pirates.
  Fact 0.43: In 2015, the listed performer had the role Helper Hue, a voice role, in Penn Zero: Part-Time Hero.
  Fact 0.44: In 2016, the listed performer had the role Ketchup Bottle / Drug Dealer / Baba Ganoush, a voice role, in Sausage Party.
  Fact 0.45: In 2016, the listed performer had the role Hugo, a voice role, in Skylanders Academy.
  Fact 0.46: From 2017 to the present, the listed performer had the role Bob, a voice role, in Puppy Dog Pals.
  Fact 0.47: The listed performer is also the creator of Puppy Dog Pals.

Retrieved items:
<item 1>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 1>

<item 2>
California Highway Patrol
They spotted the car , fell in behind , called for backup , and began the stop procedure . When the suspects ' vehicle came to a halt in the parking lot , the
driver was instructed to step out of the vehicle and spread his hands on the hood . Gore approached him and Frago moved to the passenger side . The passenger
side door suddenly swung open and the passenger sprung out , firing at Frago , who fell with two shots in his chest . The gunman , later identified as Jack
Twinning , then turned and fired once at Gore , who returned fire . In that moment the driver , Bobby Davis , turned and shot Gore twice at close range . Both
officers died instantly . When Officers James Pence and George Alleyn drove in moments later , they could not see suspects or other officers , but both
immediately came under fire . Pence put out an 11 ‐ 99 call ( `` officer down '' ) then took cover behind the passenger door . Alleyn grabbed his shotgun , and
positioned himself behind the driver - side door . Both officers were mortally wounded in the ensuing exchange , and one suspect was hit .
</item 2>

<item 3>
Harland Williams
Williams created the Disney Junior series Puppy Dog Pals that debuted in 2017 . Filmography ( edit ) Year Film Role Notes Dumb and Dumber State Trooper Down
Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny
Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr.
</item 3>

<item 4>
The Dukes of Hazzard (film)
The characters are now campus security officers , who give a warning to the Duke boys for driving too slowly . Paul Soter appears as TV newsman Rick Shakely ,
who is reporting from the Hazzard Road Rally . Charlie Finn , who played a dimwitted fast - food employee in Super Troopers , appears as a dimwitted geology
student who assists the Duke boys with the coal samples . Other notable cameos During the bar - fight scene , Indy - car driver A.J. Foyt IV appears as himself
. Bloopers roll under the ending credits , one of which features Rip Taylor interrupting the bedroom scene with Luke and the two college girls . Taylor had
previously appeared with Knoxville in Jackass : The Movie . Production ( edit ) Stunts ( edit ) Knoxville said he was initially reluctant to take on the role ,
but was persuaded by script changes and the presence of Dan Bradley as stunt coordinator and second unit for the car chase scenes . Knoxville praised him ,
saying , `` everyone in Hollywood wants Dan Bradley to shoot their car stuff '' . Locations ( edit ) The majority of the film was shot in and around Clinton ,
Louisiana .
</item 4>

<item 5>
Harland Williams
Finding a creative outlet in poetry and painting while working as a forest ranger for five years after dropping out of Sheridan College , Williams set his goals
to leave the forest as a stand - up comic and actor , soon moving to Los Angeles to pursue his newfound calling . Williams is cousin to Barenaked Ladies
keyboardist Kevin Hearn , and together they have a hobby band called The Cousins . Career ( edit ) Williams is known for his movie roles , which include Dumb
and Dumber , Freddy Got Fingered , There 's Something About Mary , Half Baked , RocketMan , Sorority Boys , Down Periscope , and Employee of The Month .
Williams starred as the security guard in the music video for the 2001 Barenaked Ladies song `` Falling for the First Time '' . He has performed his stand - up
comedy routines on Late Night with David Letterman , The Tonight Show with Jay Leno , Late Night with Conan O'Brien , HBO , Comedy Central and at comedy clubs
throughout the country . His improvisational skills led to him winning the award for ' Best Improviser ' on an episode of NBC 's prime time series Thank God You
're Here in 2007 .
</item 5>

<item 6>
Harland Williams
Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs
Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart
Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip
Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor
Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P.
</item 6>

<item 7>
The Dukes of Hazzard (film)
They are later arrested by Atlanta Police after running from campus police . Back in Hazzard , Daisy learns , with the help of Sheriff 's Deputy Enos Strate (
Michael Weston ) , that Billy Prickett has been hired by Boss Hogg to participate in the rally as a ringer . Boss Hogg then heads to Atlanta , where he informs
the Duke boys , in lock - up , that they are too late to stop him and reveals that the vote on Hogg 's proposition is at the same time as the rally , explaining
Billy Prickett 's involvement . During a transfer from detainment , Daisy helps the boys escape from the patrol car , and they speed home to try to inform the
townsfolk , escaping Atlanta Police , and the Georgia State Patrol after Bo outmaneuvers the city cops . Upon returning home , the Dukes discover that Boss Hogg
and Rosco had taken Uncle Jesse and Pauline hostage , an obvious trap for the boys , and that Billy is in on the scheme because he is ashamed of the town 's low
status .
</item 7>

<item 8>
California Highway Patrol
Peyer declined to provide a sample , nor explain why he declined to do so . Media references ( edit ) CHiPs was a fictional television drama show ( 1977 - 1983
) about the CHP , and the CHP helped out in the 1955 TV show Highway Patrol starring Broderick Crawford . Also the CHiPs TV series was made into a modern
television film called CHiPs ' 99 . A comical motion picture version of the show has been released in 2017 ( CHiPS ) but it has not been endorsed by the CHP at
all . Former CHP Commissioner Maury Hannigan was the host of the syndicated TV series Real Stories of the Highway Patrol , which aired in the mid-to - late
1990s . The show featured not just the CHP , but also the Minnesota State Patrol , the Nevada Highway Patrol , the Alaska State Troopers , the South Dakota
Highway Patrol , the Rhode Island State Police , the Massachusetts State Police , the West Virginia State Police and many other state police and highway patrol
agencies in the United State of America , and in one episode , the Royal Newfoundland Constabulary .
</item 8>

<item 9>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 9>

<item 10>
The Stanley Hotel
The Hotel has also been used as a filming location for other movies and TV shows ; most notably , as the fictional `` Hotel Danbury '' of Aspen , CO , in the
1994 film Dumb and Dumber . From 2013 to 2015 , the hotel property hosted the Stanley Film Festival , an independent horror film festival operated by the Denver
Film Society , held in early May . The festival featured screenings , panels , student competitions , audience awards and receptions . The Stanley Film Festival
was put on hiatus in 2016 , and canceled for 2017 . The historic Stanley Concert Hall serves as venue for various musical groups such as country - punk band
Murder By Death which has performed a Shining - themed series of concerts in the space . Haunted reputation ( edit ) Despite a peaceful early history , in the
years following the publication of The Shining , The Stanley Hotel has gained a reputation among paranormal investigators for frequent activity . The hotel
offers guided `` Ghost Tours '' to guests and visitors which feature spaces reputed to be exceptionally active . The hotel has also served as a location for
numerous paranormal investigation shows such as Ghost Hunters and Ghost Adventures .
</item 10>
=== END ITEM id=fc34d9261e62485f ===

=== ITEM id=2f76280ea80cee7f ===
Gold sentences and facts:
Sentence 0: The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar in 1901 CE , which started as Samvat 1958 . In Nepal , the new year begins with the first day of the month of Baishakh , which usually falls within the months of April -- May in the Gregorian calendar . The first day of the new year is passionately celebrated in a historical carnival that takes place every year in Bhaktapur , called Bisket Jatra.As before , from 2007 AD Nepal Sambat is recognized as the national calender .
  Fact 0.0: The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar.
  Fact 0.1: The Rana rulers made Vikram Samvat the official Hindu calendar in 1901 CE.
  Fact 0.2: Vikram Samvat started as Samvat 1958.
  Fact 0.3: In Nepal the new year begins on the first day of the month of Baishakh.
  Fact 0.4: Baishakh usually falls within April to May in the Gregorian calendar.
  Fact 0.5: The first day of the new year is passionately celebrated in a historical carnival called Bisket Jatra.
  Fact 0.6: The Bisket Jatra carnival takes place every year in Bhaktapur.
  Fact 0.7: As before, from 2007 AD, Nepal Sambat is recognized as the national calendar.
Sentence 1: Vikram Samvat ( Hindi : विक्रम सम्वत् , Nepali : विक्रम सम्वत् ) ( abbreviated as V.S. ( or VS ) or B.S. ( or BS ) ) ; Listen ( help info ) ) is the historical Hindu calendar of India and Nepal . It uses lunar months and solar sidereal year ( see : Vedic time keeping ) . It is used as the official calendar in Nepal .
  Fact 1.0: Vikram Samvat is written in Hindi as विक्रम सम्वत्.
  Fact 1.1: Vikram Samvat is written in Nepali as विक्रम सम्वत्.
  Fact 1.2: Vikram Samvat is abbreviated as V.S. or VS.
  Fact 1.3: Vikram Samvat is abbreviated as B.S. or BS.
  Fact 1.4: Vikram Samvat is the historical Hindu calendar of India and Nepal.
  Fact 1.5: Vikram Samvat uses lunar months.
  Fact 1.6: Vikram Samvat uses the solar sidereal year.
  Fact 1.7: Vikram Samvat is used as the official calendar in Nepal.

Retrieved items:
<item 1>
Vikram Samvat
Vikram Samvat - Wikipedia Vikram Samvat Jump to : navigation , search It has been requested that the title of this article be changed to Bikram Sambat . Please
see the relevant discussion on the discussion page . Do not move the page until the discussion has reached consensus for the change and is closed . It has been
suggested that Bikrami calendar be merged into this article . ( Discuss ) Proposed since October 2017 . Vikram Samvat ( Hindi : विक्रम सम्वत् , Nepali : विक्रम
सम्वत् ) ( abbreviated as V.S. ( or VS ) or B.S. ( or BS ) ) ; Listen ( help info ) ) is the historical Hindu calendar of India and Nepal . It uses lunar months
and solar sidereal year ( see : Vedic time keeping ) . It is used as the official calendar in Nepal . The Vikram Samvat has two alternative systems . It started
in 56 BCE in southern ( purnimanta ) and 57 -- 56 BCE in northern ( amanta ) systems of Hindu calendar .
</item 1>

<item 2>
Vikram Samvat
In India , the reformulated Saka Calendar is officially used ( although not for computing the dates of the traditional festivals ) , although in the Hindi
version of the Preamble of the Constitution of India , the date of adoption of the constitution , 26 November 1949 , is presented in Vikram Samvat ( Margsheersh
Shukla Saptami Samvat 2006 ) . There have been calls for the Vikram Samvat to replace Saka as India 's official calendar . Contents ( hide ) 1 Divisions of a
year 1.1 Lunar metrics 2 History 2.1 Vikramaditya legends 2.2 Historicity of the legends 3 Culture 4 Converting Vikram Samvat to CE 5 See also 6 References 7
Further reading Divisions of a year ( edit ) The classical Vikram Samvat uses lunar months and solar sidereal years . Because 12 months do not match a sidereal
year exactly , correctional months ( adhika māsa ) are added , or occasionally subtracted ( kshaya masa ) .
</item 2>

<item 3>
Vikram Samvat
Republican Germanic Greek Hindu Inca Macedonian Maya Haab ' Tzolk'in Muisca Pentecontad Pisan Rapa Nui Roman calendar Rumi Soviet Swedish Turkmen By specialty
Holocene ( anthropological ) Proleptic Gregorian / Proleptic Julian ( historiographical ) Darian ( Martian ) Dreamspell ( New Age ) Discordian / Pataphysical (
surreal ) Proposals Calendar reform Hanke -- Henry Permanent International Fixed Pax Positivist Symmetry454 Tranquility World New Earth Time Fictional Discworld
Greyhawk Middle - earth Stardate Star Wars ( Galactic Standard Calendar ) Displays and applications Electronic Perpetual Wall Year naming and numbering
Terminology Era Epoch Regnal name Regnal year Year zero Systems Ab urbe condita Anno Domini / Common Era Anno Mundi Assyrian Before Present Chinese Imperial
Chinese Minguo Human Era Japanese Korean Seleucid Spanish Yugas Satya Treta Dvapara Kali Vietnamese List of calendars Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Vikram_Samvat&oldid=822253746 '' Categories : Specific calendars Nepali calendar Hindu calendar Memorials to
Vikramaditya Calendar eras
</item 3>

<item 4>
Vikram Samvat
New International Encyclopedia . 1905 . ( hide ) Calendars Systems Lunar Lunisolar Solar In wide use Astronomical Bengali Chinese Ethiopian Gregorian Hebrew
Hindu Iranian Islamic ISO Unix time In more limited use Akan Armenian Assyrian Bahá'í ( Badí ' ) Balinese pawukon Balinese saka Berber Buddhist Burmese Chinese
Coptic Gaelic Germanic Heathen Georgian Hebrew Hindu or Indian Vikram Samvat Saka Igbo Iranian Jalali ( medieval ) Hijri ( modern ) Zoroastrian Islamic Fasli
Tabular Jain Japanese Javanese Korean Juche Kurdish Lithuanian Malayalam Mongolian Melanau Nanakshahi Nepal Sambat Nisg̱a'a Oromo Romanian Somali Sesotho Slavic
Slavic Native Faith Tamil Thai lunar solar Tibetan Vietnamese Xhosa Yoruba Types Runic Mesoamerican Long Count Calendar round Christian variants Julian Revised
Liturgical year Eastern Orthodox Saints Historical Attic Aztec Tonalpohualli Xiuhpohualli Babylonian Bulgar Byzantine Celtic Cham Culāsakaraj Egyptian
Florentine French Republican Germanic Greek Hindu Inca Macedonian Maya Haab ' Tzolk'in Muisca
</item 4>

<item 5>
Vikram Samvat
It occurs in mid-April each year , and coincides with the traditional new year in Assam , Bengal , Burma , Cambodia , Kerala , Kashmir , Manipur , Odisha ,
Punjab , Sri Lanka , Tamil Nadu and Thailand . In addition to Nepal , the Vikram Samvat calendar is also recognized in North and East India , and in Gujarat
among Hindus . In Buddhist communities , the month of Baishakh is associated with Vesak or Buddha 's Birthday . It commemorates the birth , Enlightenment and
passing of Gautama Buddha on the first full moon day in May , except in a leap year when the festival is held in June . Although this festival is not held on
the same day as Pahela Baishakh , the holidays typically fall in the same month ( Baishakh ) of the Bengali , Hindu , and Theravada Buddhist calendars , and are
related historically through the spread of Hinduism and Buddhism in South Asia . In Gujarat , the second day of Diwali is celebrated as the first day of the
Vikram Samvat calendar which is the first day of the month Kartik .
</item 5>

<item 6>
Vikram Samvat
According to Rudolf Hoernlé , the king responsible for this change was Yashodharman : Hoernlé also believed that he conquered Kashmir , and is same person as
the `` Harsha Vikramaditya '' mentioned in Kalhana 's Rajatarangini . Earlier , some scholars believed that the Vikrama Samavat corresponded to the Azes era of
the Indo - Scythian ( Śaka ) king King Azes . However , this was disputed by Robert Bracey following discovery of an inscription of Vijayamitra , which is dated
in two eras . The theory seems to be now thoroughly discredited by Falk and Bennett , who place the inception of the Azes era in 47 -- 46 BCE . Culture ( edit )
The traditional New Year of Vikram Samvat is one of the many festivals of Nepal , marked by parties , family gatherings , the exchange of good wishes , and
participation in rituals to ensure good fortune in the coming year .
</item 6>

<item 7>
Vikram Samvat
Converting Vikram Samvat to CE ( edit ) While an approximate conversion can be made by subtracting 56 or 57 , a precise computation requires identification of
the version of Vikam Samvat that was used : Year expired ( default ) or current Kartikadi ( older or western India ) or Chaitradi ( modern Northern India ) .
Kartikadi starts 7 months later in Kartika . Until 12th century Kartikadi year was common even in North India . Months Amanta ( south ) or Purminata ( North ) .
Amanta Chaitra Krishna Paksha is Purminata Vaishakh Krishna Paksha . For the Shukla Paksh , the month is the same . Software named Pancanga is available to make
the calculations . Thus the date Vikram 1208 ( Kartikadi , expired ) , Vaishakh Krishna 5 ( Purnimanta ) , Thursday , is equivalent to Vikram 1209 ( Chaitradi ,
expired ) , Chaitra Krishna 5 ( Amanta ) , and to Thursday March 27 , 1152 .
</item 7>

<item 8>
Vikram Samvat
BRILL . p. 103 . ISBN 90 - 04 - 18566 - 6 . Jump up ^ Falk and Bennett ( 2009 ) , pp. 197 - 215 . Jump up ^ `` Gujarat CM to exchange Diwali - New Year
greetings with people '' . 19 October 2014 . Retrieved 24 October 2014 . Jump up ^ Pancanga ( version 3.14 ) Jump up ^ On a Jain Statue in the Horniman Museum ,
F. Kielhorn , Journal of the Royal Asiatic Society of Great Britain and Ireland , 1898 , p. 101 - 102 Further reading ( edit ) Wikisource has the text of the
1911 Encyclopædia Britannica article Vikramaditya . Harry Falk and Chris Bennett ( 2009 ) . `` Macedonian Intercalary Months and the Era of Azes . '' Acta
Orientalia 70 , pp. 197 -- 215 . `` The Dynastic Art of the Kushan '' , John Rosenfield . `` Samvat '' . New International Encyclopedia . 1905 . ( hide )
Calendars Systems
</item 8>

<item 9>
Vikram Samvat
The Shukla Paksha in both systems coincides , most festivals occur in the Shukla Paksha . The era is named after King Vikramaditya of India . The lunisolar
Vikram Samvat calendar is 56.7 years ahead ( in count ) of the solar Gregorian calendar . For example , the year 2074 VS began in 2017 CE and will end in 2018
CE . The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar in 1901 CE , which started as Samvat 1958 . In Nepal , the new year begins with the
first day of the month of Baishakh , which usually falls within the months of April -- May in the Gregorian calendar . The first day of the new year is
passionately celebrated in a historical carnival that takes place every year in Bhaktapur , called Bisket Jatra.As before , from 2007 AD Nepal Sambat is
recognized as the national calender .
</item 9>

<item 10>
Vikram Samvat
See also ( edit ) Bikrami calendar Hindu calendars Vira Nirvana Samvat References ( edit ) Jump up ^ Richard Salomon ( 1398 ) . Indian Epigraphy : A Guide to
the Study of Inscriptions in Sanskrit , Prakrit , and the other Indo - Aryan Languages . Oxford University Press . pp. 181 -- 183 . ISBN 978 - 0 - 19 - 535666 -
3 . ^ Jump up to : Ashvini Agrawal ( 1989 ) . Rise and Fall of the Imperial Guptas . Motilal Banarsidass Publ . pp. 174 -- 175 . ISBN 978 - 81 - 208 - 0592 - 7
. Jump up ^ Encyclopedia of New Year 's Holidays Worldwide , William D. Crump , Publisher McFarland , 2014 p. 38 Jump up ^ `` Vikram Samvat should be declared
national calendar '' . The Free Press Journal . 15 February 2012 . Archived from the original on 26 April 2014 . Retrieved 28 March 2012 .
</item 10>
=== END ITEM id=2f76280ea80cee7f ===
