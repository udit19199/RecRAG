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

=== ITEM id=35f56622d8f37d5f ===
Gold sentences and facts:
Sentence 0: The nineteenth season of Law & Order : Special Victims Unit premiered on September 27 , 2017 .
  Fact 0.0: The nineteenth season of Law & Order: Special Victims Unit premiered.
  Fact 0.1: The nineteenth season of Law & Order: Special Victims Unit premiered on September 27, 2017.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
ET for the first nine episodes , from September 20 through November 29 , 1999 . It was then shifted to Friday nights at 10 p.m. ET on January 7 , 2000 , and
remained in that time slot through the end of season four on May 16 , 2003 . Beginning with the season five premiere on September 23 , 2003 , SVU moved to
Tuesday nights at 10 : 00 p.m. ET . NBC airs repeats on Saturday nights at varying times and previous episodes are shown on the USA Network on varying days in
marathon blocks With the season eleven premiere on September 23 , 2009 , the series vacated its Tuesday 10 p.m. ET slot because NBC began a prime - time
weeknight Jay Leno series . The new time slot became Wednesday nights at 9 : 00 p.m. ET on NBC , with CTV still airing SVU on Tuesdays at 10 : 00 in Canada .
After the 2010 Winter Olympics on March 3 , 2010 , the time slot for SVU changed again to Wednesdays at 10 p.m.
</item 1>

<item 2>
*Law & Order: Special Victims Unit* (SVU) is a New York City police procedural and legal crime drama created by Dick Wolf. Premiering on NBC on September 20,
1999, it became the second series in the *Law & Order* franchise. Episodes often draw on real crimes. Mariska Hargitay won a 2006 Emmy for Outstanding Lead
Actress in a Drama Series.
</item 2>

<item 3>
*Law & Order: SVU* shifted between 9 and 10 p.m. over several seasons: it moved to 9 p.m. in season 12 to lead into *Law & Order: LA*, returned to 10 p.m. in
January 2011 after that spinoff was pulled, then moved back to 9 p.m. for season 14 in September 2012, where it remained. In 2007, Studio 2B acquired rights to
make a
</item 3>

<item 4>
Law & Order: Special Victims Unit
2017 ( 2017 - 09 - 27 ) TBA TBA TBA Reception ( edit ) U.S. television Ratings ( edit ) Law & Order : Special Victims Unit premiered on Monday , September 20 ,
1999 . After nine episodes , the show was moved to Friday nights , where it found its audience and following its first and second seasons became a Top 20 -
overall show , overcoming the `` Friday night death slot '' phenomenon . Beginning with the season five , the show aired on Tuesdays to compete with CBS '
Judging Amy and ABC 's NYPD Blue . In its later years , Law & Order : Special Victims Unit consistently outperformed Law & Order in the Nielsen ratings for
first run episodes until the latter 's cancellation in 2010 . In 2016 , a New York Times study of the 50 TV shows with the most Facebook Likes found that SVU 's
popularity was `` atypical : generally slightly more popular in rural areas and the Black Belt , but largely restricted to the eastern half of the country . It
's most popular in Albany , N.Y. ; least in Colorado and Utah '' .
</item 4>

<item 5>
*Law & Order: SVU* aired Monday at 9 p.m. from its 1999 premiere through November, then Friday at 10 p.m. until May 2003. It moved to Tuesday at 10 p.m. in
September 2003, then Wednesday at 9 p.m. in September 2009, and Wednesday at 10 p.m. in March 2010. NBC and USA aired repeats; the series briefly aired on Syfy
in 2006
</item 5>

<item 6>
Law & Order: Special Victims Unit
1999 ( 1999 - 09 - 20 ) -- present Chronology Related shows Law & Order franchise External links Website Law & Order : Special Victims Unit ( often abbreviated
to Law & Order : SVU or just SVU ) is an American police procedural , legal , crime drama television series set in New York City , where it is also primarily
produced . In the style of the original Law & Order , episodes are often `` ripped from the headlines '' or loosely based on real crimes that have received
media attention . Created by Dick Wolf , the series premiered on NBC on September 20 , 1999 , as the second series in Wolf 's successful Law & Order franchise .
Law & Order : Special Victims Unit has been nominated for and won numerous awards , including the 2006 Emmy for Outstanding Lead Actress in a Drama Series for
Mariska Hargitay , the first , and to date only , Emmy to be received by a regular on any Law & Order series . It is the current longest running scripted
non-animated U.S.
</item 6>

<item 7>
Law & Order: Special Victims Unit
ET , where it stayed until the twelfth season . In the 12th season , SVU moved back to 9 : 00 p.m. to lead in the newest Law & Order spinoff , Law & Order : LA
, until it was pulled from the network in January 2011 to be retooled . SVU moved back to 10 : 00 p.m. on January 12 , 2011 , until the end of the 13th season .
With season 14 , SVU moved back to 9 : 00 p.m. after a two - hour season premiere event on September 26 , 2012 , a position the show has held since . Russian
adaptation ( edit ) In 2007 , the Russian production company Studio 2B purchased the rights to create an adaptation of Law & Order : Special Victims Unit for
Russian television . Titled Закон и порядок : отдел оперативных расследований ( Law & Order : Division of Field Investigation ) , the series stars Alisa Bogart
as Major Olga Bobrova .
</item 7>

<item 8>
Law & Order: Special Victims Unit
The first episode , `` Payback '' , premiered on NBC on September 20 , 1999 . Executive producers ( edit ) Executive producer Neal Baer left Law & Order : SVU
as showrunner at the end of season twelve , after eleven years ( seasons 2 -- 12 ) on the show , in order to sign a three - year deal with CBS Studios . Baer
was replaced by former Law & Order : Criminal Intent showrunner Warren Leight . In March 2015 , it was announced that Warren Leight signed a three - year deal
with Sony Pictures Television , that will allow him to work on SVU one more season , its seventeenth . Leight joined the show in season thirteen . It was
announced on March 10 , 2016 that original Law & Order veteran producer Rick Eid would take Leight 's place as showrunner starting in season 18 . Creator Dick
Wolf commented to The Hollywood Reporter , `` I 'm extremely pleased that Rick had decided to rejoin the family and hope that he will be here for years to come
. ''
</item 8>

<item 9>
The latest five episodes of *Law & Order: SVU* are free on NBC.com and the NBC app; all episodes are available to buy on platforms such as iTunes and Amazon
Video. In Australia, Network Ten airs new episodes Thursdays, with Season 19 premiering October 5, 2017, at 9:40 p.m. Channel Nine began showing Season 1 in
August 2017 but stopped after one episode; Ten’s digital channel One had aired early-season repeats.
</item 9>

<item 10>
Law & Order: Special Victims Unit
The show also briefly ran on Syfy in 2006 . International broadcast ( edit ) In Australia , Law & Order : SVU airs on Network Ten and Australia 's cable channel
Universal Channel ( formerly aired on TV1 which ceased to broadcast in December 2013 ) . In the UK `` Law & Order : SVU '' originally aired on the Hallmark
Channel ( which in 2012 became the Universal Channel ) with a terrestrial TV airing on Channel 5 in 2016 which then switched to Five USA . The UK is two and
half weeks behind the US . In Brazil , Law & Order : SVU also airs on Universal Channel , with weekdays episodes from past seasons and simultaneously , two
episodes after the original airing in the US . Law & Order : SVU is broadcast at 9.30 pm on Tuesdays on TV3 in New Zealand . Series 18 premiered on 11 April
2017 . References ( edit ) Notes ( edit ) Jump up ^ SVU Writers Room ( @ SVUWritersRoom ) ( 7 July 2016 ) . `` First cast read - through of # SVUSeason18 '' (
Tweet ) -- via Twitter .
</item 10>
=== END ITEM id=35f56622d8f37d5f ===

=== ITEM id=ac7ecfca88b50844 ===
Gold sentences and facts:
Sentence 0: All I Want for Christmas Is You , a Night of Joy and Festivity is the first concert residency by American singer Mariah Carey . Performed annually at the Beacon Theatre in Manhattan , New York , the residency began on December 15 , 2014 , and will conclude on December 22 , 2017 , after completing six legs and thirty - four shows . Billboard revealed that the six dates had sold out with a total of 16,196 tickets while grossing $1.5 million during the first leg of the residency . The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You , with additional non-holiday tracks . The show also included `` a cast of forty , including a gospel choir , a clutch of carolers , several acrobats and a hyperactive round of kid singer / dancers . ''
  Fact 0.0: 'All I Want for Christmas Is You, a Night of Joy and Festivity' is the first concert residency by American singer Mariah Carey.
  Fact 0.1: The residency is performed annually at the Beacon Theatre in Manhattan, New York.
  Fact 0.2: The residency began on December 15, 2014.
  Fact 0.3: The residency will conclude on December 22, 2017.
  Fact 0.4: The residency completes six legs and thirty-four shows.
  Fact 0.5: Billboard revealed that the six dates had sold out with a total of 16,196 tickets.
  Fact 0.6: The six dates grossed $1.5 million during the first leg of the residency.
  Fact 0.7: The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You.
  Fact 0.8: The main set list for the show included additional non-holiday tracks.
  Fact 0.9: The show included a cast of forty.
  Fact 0.10: The show's cast included a gospel choir, a clutch of carolers, several acrobats, and a hyperactive round of kid singer/dancers.

Retrieved items:
<item 1>
All I Want for Christmas Is You, a Night of Joy and Festivity
All I Want for Christmas is You , a Night of Joy and Festivity - wikipedia All I Want for Christmas is You , a Night of Joy and Festivity Jump to : navigation ,
search All I Want for Christmas Is You , a Night of Joy and Festivity Residency by Mariah Carey Location North America Europe Venue Various Associated albums
Merry Christmas & Merry Christmas II You Start date December 15 , 2014 ( 2014 - 12 - 15 ) End date December 22 , 2017 ( 2017 - 12 - 22 ) Legs 6 No.
</item 1>

<item 2>
All I Want for Christmas Is You, a Night of Joy and Festivity
of shows 6 in 2014 8 in 2015 9 in 2016 11 in 2017 34 in total Attendance 65,195 ( 93 % ) Box office $7,335,589 Mariah Carey concert chronology The Elusive
Chanteuse Show ( 2014 ) All I Want for Christmas Is You , a Night of Joy & Festivity ( 2014 - 17 ) Number 1 to Infinity ( 2015 -- 17 ) All I Want for Christmas
Is You , a Night of Joy and Festivity is the first concert residency by American singer Mariah Carey . Performed annually at the Beacon Theatre in Manhattan ,
New York , the residency began on December 15 , 2014 , and will conclude on December 22 , 2017 , after completing six legs and thirty - four shows . Billboard
revealed that the six dates had sold out with a total of 16,196 tickets while grossing $1.5 million during the first leg of the residency .
</item 2>

<item 3>
All I Want for Christmas Is You, a Night of Joy and Festivity
195 / 74,089 ( 93 % ) $7,335,589 Cancelled shows ( edit ) Date Venue City Country Reason 2017 November 17 , 2017 Caesars Windsor Windsor Canada `` Artist
illness '' ( Upper respiratory infection ) November 18 , 2017 Chicago Theatre Chicago United States November 20 , 2017 Sands Bethlehem Event Center Bethlehem
November 21 , 2017 Seneca Niagara Events Center Niagara Falls November 22 , 2017 Hard Rock Rocksino Northfield Park Cleveland November 24 , 2017 MGM National
Harbor Washington , D.C. November 25 , 2017 Borgata Hotel Casino & Spa Atlantic City November 28 , 2017 Beacon Theatre New York City December 1 , 2017
References ( edit ) Jump up ^ Lopez , Francisco ( October 31 , 2014 ) . `` Mariah Carey 's ' All I Want For Christmas Is You ' performance at the Beacon NYC ''
. AXS.com . Retrieved January 28 , 2015 . Jump up ^ Gamboa , Glenn ( October 31 , 2014 ) .
</item 3>

<item 4>
Graph query row:
entity: All I Want for Christmas Is You, a Night of Joy and Festivity
relationship: LOCATED_IN
related: Beacon Theatre
related_labels: ['__KGBuilder__', '__Entity__', 'Place']
date: None
kind: venue
role: None
relationship_role: None
context: None
relationship_detail: None
relation: None
fact: None
result: None
outcome: None
tenure: None
period: None
term: None
</item 4>

<item 5>
All I Want for Christmas Is You, a Night of Joy and Festivity
The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You , with additional non-holiday tracks . The show also included
`` a cast of forty , including a gospel choir , a clutch of carolers , several acrobats and a hyperactive round of kid singer / dancers . '' Contents ( hide ) 1
Background and development 2 Critical reception 3 Commercial performance 4 Set list 5 Residency dates 5.1 Cancelled shows 6 References Background and
development ( edit ) Mariah Carey released her fourteenth studio album , titled Me . I Am Mariah ... The Elusive Chanteuse , on May 27 , 2014 . It debuted at
number three on the Billboard 200 album chart with opening week sales of 58,000 copies in the United States , and number one on the Top R&B / Hip - Hop Albums
chart . Carey embarked on her eighth concert tour called The Elusive Chanteuse Show in October and November later that year , whereby she performed concerts
throughout East Asia , South East Asia and Australasia .
</item 5>

<item 6>
All I Want for Christmas Is You, a Night of Joy and Festivity
Jump up ^ http://www.torontosun.com/2017/07/19/mariah-carey-brings-christmas-concert-series-to-las-vegas Jump up ^
http://windsorstar.com/news/local-news/mariah-carey-cancels-caesars-windsor-show Jump up ^
http://www.etonline.com/mariah-carey-cancels-first-shows-christmas-concert-due-respiratory-infection-91147 Jump up ^
http://www.billboard.com/articles/columns/pop/8038463/mariah-carey-cancels-shows-christmas-tour Jump up ^
http://metro.co.uk/2017/11/25/mariah-carey-cancels-more-christmas-tour-shows-for-a-few-extra-days-rest-7107422/ Jump up ^ http://www.eonline.
</item 6>

<item 7>
All I Want for Christmas Is You, a Night of Joy and Festivity
wikipedia.org/w/index.php?title=All_I_Want_for_Christmas_Is_You,_a_Night_of_Joy_and_Festivity&oldid=819618785 '' Categories : 2014 concert residencies 2015
concert residencies 2016 concert residencies 2017 concert residencies Mariah Carey concert residencies Christmas music Hidden categories : Pages using div col
with deprecated parameters Talk Contents About Wikipedia Español Bahasa Indonesia Português Edit links This page was last edited on 10 January 2018 , at 10 : 27
. Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of
Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
All I Want for Christmas Is You, a Night of Joy and Festivity
Jump up ^ Gamboa , Glenn ( October 31 , 2014 ) . `` Mariah Carey sets holiday shows at Beacon Theatre '' . Newsday . Retrieved January 28 , 2015 . ^ Jump up to
: Farber , Jim ( December 15 , 2014 ) . `` Mariah Carey overdoes herself at wavering Beacon Theatre show : concert review '' . Daily News . New York . Retrieved
January 28 , 2015 . ^ Jump up to : Stern , Bradley ( August 18 , 2014 ) . `` Mariah Carey Announces ' The Elusive Chanteuse Show ' : See The First Tour Dates ''
. Idolator . Archived from the original on October 6 , 2017 . Retrieved October 6 , 2017 . ^ Jump up to : Brandle , Lars ( September 18 , 2014 ) . `` Mariah
Carey Announces ' Elusive Chanteuse ' Tour of Australia '' . Billboard . Archived from the original on October 6 , 2017 .
</item 8>

<item 9>
All I Want for Christmas Is You, a Night of Joy and Festivity
eonline.com/news/896163/mariah-carey-further-postpones-christmas-tour-over-health-issues Book : Merry Christmas II You ( hide ) Mariah Carey Albums Songs Awards
and nominations Singles Videography Performances Studio albums Mariah Carey Emotions Music Box Merry Christmas Daydream Butterfly Rainbow Charmbracelet The
Emancipation of Mimi E = MC2 Memoirs of an Imperfect Angel Merry Christmas II You Me . I Am Mariah ...
</item 9>

<item 10>
All I Want for Christmas Is You, a Night of Joy and Festivity
I Am Mariah ... The Elusive Chanteuse Live albums MTV Unplugged Compilation albums # 1 's Greatest Hits The Remixes The Ballads The Essential Mariah Carey # 1
to Infinity Soundtrack albums Glitter Video albums The First Vision MTV Unplugged + 3 Here Is Mariah Carey Fantasy : Mariah Carey at Madison Square Garden
Around the World # 1 's The Adventures of Mimi Television VH1 Divas American Idol A Christmas Melody Mariah 's World Concert tours Music Box Tour Daydream World
Tour Butterfly World Tour Rainbow World Tour Charmbracelet World Tour The Adventures of Mimi Angels Advocate Tour The Elusive Chanteuse Show The Sweet Sweet
Fantasy Tour All the Hits Tour The # 1 's Tour Concert residencies Live at the Pearl All I Want for Christmas Is You , a Night of Joy and Festivity # 1 to
Infinity Fragrances M by Mariah Carey Luscious Pink Forever Companies Monarc Entertainment Butterfly MC Records Related topics Tommy Mottola Nick Cannon Lionel
Cole James Packer VH1 Divas Book Portal Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=ac7ecfca88b50844 ===

=== ITEM id=7b8281b9d3848dcd ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 `` Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017 - 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 ) 705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 - 01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD
  Fact 0.0: Episode 107 overall is titled 'Watch Me'.
  Fact 0.1: Episode 107 overall was directed by Jann Turner.
  Fact 0.2: Episode 107 overall was written by Shonda Rhimes.
  Fact 0.3: Episode 107 overall originally aired on October 5, 2017.
  Fact 0.4: Episode 107 overall has production code 701.
  Fact 0.5: Episode 107 overall had 5.52 million U.S. viewers.
  Fact 0.6: Episode 108 overall is titled 'Pressing the Flesh'.
  Fact 0.7: Episode 108 overall was directed by Tony Goldwyn.
  Fact 0.8: Episode 108 overall was written by Matt Byrne.
  Fact 0.9: Episode 108 overall originally aired on October 12, 2017.
  Fact 0.10: Episode 108 overall has production code 702.
  Fact 0.11: Episode 108 overall had 5.00 million U.S. viewers.
  Fact 0.12: Episode 109 overall is titled 'Day 101'.
  Fact 0.13: Episode 109 overall was directed by Scott Foley.
  Fact 0.14: Episode 109 overall was written by Zahir McGhee.
  Fact 0.15: Episode 109 overall originally aired on October 19, 2017.
  Fact 0.16: Episode 109 overall has production code 703.
  Fact 0.17: Episode 109 overall had 4.70 million U.S. viewers.
  Fact 0.18: Episode 110 overall is titled 'Lost Girls'.
  Fact 0.19: Episode 110 overall was directed by Nicole Rubio.
  Fact 0.20: Episode 110 overall was written by Ameni Rozsa and Austin Guzman.
  Fact 0.21: Episode 110 overall originally aired on October 26, 2017.
  Fact 0.22: Episode 110 overall has production code 704.
  Fact 0.23: Episode 110 overall had 4.88 million U.S. viewers.
  Fact 0.24: Episode 111 overall is episode 5 of its season and is titled 'Adventures in Babysitting'.
  Fact 0.25: Episode 111 overall was directed by Oliver Bokelberg.
  Fact 0.26: Episode 111 overall was written by Serveriano Canales and Tia Napolitano.
  Fact 0.27: Episode 111 overall originally aired on November 2, 2017.
  Fact 0.28: Episode 111 overall has production code 705.
  Fact 0.29: Episode 111 overall had 4.89 million U.S. viewers.
  Fact 0.30: Episode 112 overall is episode 6 of its season and is titled 'Vampires and Bloodsuckers'.
  Fact 0.31: Episode 112 overall was directed by Jann Turner.
  Fact 0.32: Episode 112 overall was written by Chris Van Dusen and Tia Napolitano.
  Fact 0.33: Episode 112 overall originally aired on November 9, 2017.
  Fact 0.34: Episode 112 overall has production code 706.
  Fact 0.35: Episode 112 overall had 5.00 million U.S. viewers.
  Fact 0.36: Episode 113 overall is episode 7 of its season and is titled 'Something Borrowed'.
  Fact 0.37: Episode 113 overall was directed by Sharat Raju.
  Fact 0.38: Episode 113 overall was written by Mark Fish.
  Fact 0.39: Episode 113 overall originally aired on November 16, 2017.
  Fact 0.40: Episode 113 overall has production code 707.
  Fact 0.41: Episode 113 overall had 4.97 million U.S. viewers.
  Fact 0.42: Episode 114 overall is episode 8 of its season and is titled 'Robin'.
  Fact 0.43: Episode 114 overall was directed by Daryn Okada.
  Fact 0.44: Episode 114 overall was written by Juan Carlos Fernandez.
  Fact 0.45: Episode 114 overall originally aired on January 18, 2018.
  Fact 0.46: Episode 114 overall has production code 709.
  Fact 0.47: Episode 114 overall had 5.17 million U.S. viewers.
  Fact 0.48: Episode 115 overall is episode 9 of its season and is titled 'Good People'.
  Fact 0.49: Episode 115 overall was directed by Nzingha Stewart.
  Fact 0.50: Episode 115 overall was written by Shonda Rhimes, Jess Brownell and Nicholas Nardini.
  Fact 0.51: Episode 115 overall originally aired on January 25, 2018.
  Fact 0.52: Episode 115 overall has production code 708.
  Fact 0.53: Episode 115 overall has U.S. viewership listed as TBD.
  Fact 0.54: Episode 116 overall is episode 10 of its season and is titled 'The People v. Olivia Pope'.
  Fact 0.55: Episode 116 overall has its director listed as TBA.
  Fact 0.56: Episode 116 overall has its writer listed as TBA.
  Fact 0.57: Episode 116 overall originally aired on February 1, 2018.
  Fact 0.58: Episode 116 overall has production code 710.
  Fact 0.59: Episode 116 overall has U.S. viewership listed as TBD.
  Fact 0.60: Episode 117 overall is episode 11 of its season and is titled 'Army of One'.
  Fact 0.61: Episode 117 overall has its director listed as TBA.
  Fact 0.62: Episode 117 overall has its writer listed as TBA.
  Fact 0.63: Episode 117 overall originally aired on February 8, 2018.
  Fact 0.64: Episode 117 overall has production code 711.
  Fact 0.65: Episode 117 overall has U.S. viewership listed as TBD.
  Fact 0.66: Episode 118 overall is episode 12 of its season.
  Fact 0.67: Episode 118 overall has its title, director, and writer listed as TBA.
  Fact 0.68: Episode 118 overall originally aired on March 1, 2018.
  Fact 0.69: Episode 118 overall has its production code listed as TBA.
  Fact 0.70: Episode 118 overall has U.S. viewership listed as TBD.
Sentence 1: 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD
  Fact 1.0: Episode 116 overall is episode 10 of its season and is titled 'The People v. Olivia Pope'.
  Fact 1.1: Episode 116 overall has its director listed as TBA.
  Fact 1.2: Episode 116 overall has its writer listed as TBA.
  Fact 1.3: Episode 116 overall originally aired on February 1, 2018.
  Fact 1.4: Episode 116 overall has production code 710.
  Fact 1.5: Episode 116 overall has U.S. viewership listed as TBD.

Retrieved items:
<item 1>
List of Scandal episodes
TV by the Numbers . Retrieved January 22 , 2018 . Jump up ^ `` ( # 708 ) `` Good People '' `` . The Futon Critic . Retrieved January 24 , 2018 . ^ Jump up to :
`` Shows A-Z - scandal on abc '' . The Futon Critic . Retrieved January 24 , 2018 . Jump up ^ `` ' Scandal ' Sets Darby Stanchfield - Directed Web Series Ahead
of Season 6 '' . The Hollywood Reporter . January 18 , 2017 . External links ( edit ) Official website List of Scandal episodes on IMDb ( hide ) Scandal
Episodes Season 1 `` Sweet Baby '' Season 2 `` Nobody Likes Babies '' `` Whiskey Tango Foxtrot '' `` Boom Goes the Dynamite '' `` Top of the Hour '' `` Snake in
the Garden '' `` Molly , You in Danger , Girl '' `` Seven Fifty - Two '' Season 3 `` It 's Handled '' `` Guess Who 's Coming to Dinner '' `` Mrs.
</item 1>

<item 2>
List of Scandal episodes
List of Scandal episodes - wikipedia List of Scandal episodes Jump to : navigation , search Scandal is an American political thriller television series created
by Shonda Rhimes , which premiered on ABC on April 5 , 2012 . Kerry Washington stars as Olivia Pope , a former White House Communications Director who leaves to
start her own crisis management firm , Pope and Associates , where she works to keep the secrets and protect the public images of the country 's most powerful
and elite . On staff at the new firm are fresh - faced lawyer Quinn Perkins ( Katie Lowes ) ; smooth talking litigator Harrison Wright ( Columbus Short ) ;
investigator Abby Whelan ( Darby Stanchfield ) ; and computer hacker with a CIA past Huck ( Guillermo Diaz ) . The show also focuses on President of the United
States Fitzgerald Grant ( Tony Goldwyn ) ; First Lady Mellie Grant ( Bellamy Young ) ; White House Chief of Staff Cyrus Beene ( Jeff Perry ) ; Vice President
and eventual presidential candidate Sally Langston ( Kate Burton ) ; U.S.
</item 2>

<item 3>
List of Scandal episodes
May 26 , 2016 . Retrieved May 27 , 2016 . Jump up ^ `` Final 2016 - 17 TV Rankings : ' Sunday Night Football ' Winning Streak Continues '' . Deadline Hollywood
. May 26 , 2017 . Retrieved May 26 , 2017 . Jump up ^ Ausiello , Michael ( June 26 , 2017 ) . `` Scandal Farewell Update : Final Season Episode Count Revealed
'' . TVLine . Retrieved July 28 , 2017 . Jump up ^ Gelman , Vlada ( January 8 , 2018 ) . `` Quantico Season 3 and Shondaland 's For the People Get ABC Premiere
Dates '' . TVLine . Retrieved January 8 , 2018 . Jump up ^ Bibel , Sara ( April 6 , 2012 ) . `` Thursday Final Ratings : ' Big Bang Theory ' , ' American Idol '
, ' Person of Interest ' , ' Missing ' , ' Up All Night ' Adjusted Up ; ' Scandal ' Adjusted Down '' . TV by the Numbers .
</item 3>

<item 4>
List of Scandal episodes
Attorney David Rosen ( Joshua Malina ) ; Head of NSA , Jake Ballard ( Scott Foley ) ; chairwoman Elizabeth North ( Portia de Rossi ) ; activist Marcus Walker (
Cornelius Smith Jr . ) ; and head of B613 Eli Pope ( Joe Morton ) . As of January 18 , 2018 , 114 episodes of Scandal have aired , including one special . On
February 10 , 2017 , the show was renewed for a seventh season consisting of 18 episodes . It was later announced it will be the final season . Contents ( hide
) 1 Series overview 2 Episodes 2.1 Season 1 ( 2012 ) 2.2 Season 2 ( 2012 -- 13 ) 2.3 Season 3 ( 2013 -- 14 ) 2.4 Season 4 ( 2014 -- 15 ) 2.5 Season 5 ( 2015 --
16 ) 2.6 Season 6 ( 2017 ) 2.7 Season 7 ( 2017 -- 18 ) 3 Specials 4 Webisodes 4.
</item 4>

<item 5>
List of Scandal episodes
2014 ( 2014 - 04 - 17 ) 318 10.57 Season 4 ( 2014 -- 15 ) ( edit ) Main article : Scandal ( season 4 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 48 `` Randy , Red , Superfreak and Julia '' Tom Verica Shonda Rhimes September 25 , 2014 ( 2014 - 09 -
25 ) 401 11.96 49 `` The State of the Union '' Allison Liddi - Brown Heather Mitchell October 2 , 2014 ( 2014 - 10 - 02 ) 402 10.34 50 `` Inside the Bubble ''
Randy Zisk Matt Byrne October 9 , 2014 ( 2014 - 10 - 09 ) 403 9.52 51 `` Like Father , Like Daughter '' Paul McCrane Mark Fish October 16 , 2014 ( 2014 - 10 -
16 ) 404A 9.
</item 5>

<item 6>
List of Scandal episodes
07 28 21 `` Any Questions ? '' Mark Tinker Matt Byrne May 9 , 2013 ( 2013 - 05 - 09 ) 221 8.87 29 22 `` White Hat 's Back On '' Tom Verica Shonda Rhimes May 16
, 2013 ( 2013 - 05 - 16 ) 222 9.12 Season 3 ( 2013 -- 14 ) ( edit ) Main article : Scandal ( season 3 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 30 `` It 's Handled '' Tom Verica Shonda Rhimes October 3 , 2013 ( 2013 - 10 - 03 ) 301 10.52 31 ``
Guess Who 's Coming to Dinner '' Allison Liddi - Brown Heather Mitchell October 10 , 2013 ( 2013 - 10 - 10 ) 302 9.01 32 `` Mrs. Smith Goes to Washington ''
Jeannot Szwarc Matt Byrne October 17 ,
</item 6>

<item 7>
List of Scandal episodes
2012 ( 2012 - 05 - 17 ) 107 7.33 Season 2 ( 2012 -- 13 ) ( edit ) Main article : Scandal ( season 2 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 8 `` White Hat 's Off '' Tom Verica Jenna Bans September 27 , 2012 ( 2012 - 09 - 27 ) 201 6.74 9 `` The
Other Woman '' Stephen Cragg Heather Mitchell October 4 , 2012 ( 2012 - 10 - 04 ) 202 6.56 10 `` Hunting Season '' Ron Underwood Matt Byrne October 18 , 2012 (
2012 - 10 - 18 ) 203 6.17 11 `` Beltway Unbuckled '' Mark Tinker Mark Fish October 25 , 2012 ( 2012 - 10 - 25 ) 204 6.11 12 5 `` All Roads Lead to Fitz '' Steve
Robin Raamla Mohamed November 8 ,
</item 7>

<item 8>
List of Scandal episodes
Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01
) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD Specials (
edit ) No . Title Narrator Aired between Original air date U.S.
</item 8>

<item 9>
List of Scandal episodes
2015 ( 2015 - 09 - 24 ) May 12 , 2016 ( 2016 - 05 - 12 ) 16 10.68 6 16 January 26 , 2017 ( 2017 - 01 - 26 ) May 18 , 2017 ( 2017 - 05 - 18 ) 30 8.16 7 18
October 5 , 2017 ( 2017 - 10 - 05 ) April 19 , 2018 ( 2018 - 04 - 19 ) N / A N / A Episodes ( edit ) Season 1 ( 2012 ) ( edit ) Main article : Scandal ( season
1 ) No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) `` Sweet Baby '' Paul McGuigan Shonda Rhimes
April 5 , 2012 ( 2012 - 04 - 05 ) 101 7.33 `` Dirty Little Secrets '' Roxann Dawson Heather Mitchell April 12 ,
</item 9>

<item 10>
List of Scandal episodes
2017 ( 2017 - 05 - 18 ) 616 5.23 Season 7 ( 2017 -- 18 ) ( edit ) Main article : Scandal ( season 7 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 ``
Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017
- 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.
</item 10>
=== END ITEM id=7b8281b9d3848dcd ===

=== ITEM id=a7300cd6a8e0e822 ===
Gold sentences and facts:
Sentence 0: The Sentencing Reform Act , part of the Comprehensive Crime Control Act of 1984 , was a U.S. federal statute intended to increase consistency in United States federal sentencing . It established the United States Sentencing Commission . It also abolished federal parole , except for persons convicted under federal law before November 1 , 1987 , persons convicted under District of Columbia law , `` transfer treaty '' inmates , persons who violated military law who are in federal civilian prisons , and persons who are defendants in state cases and who are under the U.S. Marshals Service Witness Protection Program .
  Fact 0.0: The Sentencing Reform Act was part of the Comprehensive Crime Control Act of 1984.
  Fact 0.1: The Sentencing Reform Act was a U.S. federal statute.
  Fact 0.2: The Sentencing Reform Act was intended to increase consistency in United States federal sentencing.
  Fact 0.3: The Sentencing Reform Act established the United States Sentencing Commission.
  Fact 0.4: The Sentencing Reform Act abolished federal parole.
  Fact 0.5: The abolition of federal parole did not apply to persons convicted under federal law before November 1, 1987.
  Fact 0.6: The abolition of federal parole did not apply to persons convicted under District of Columbia law.
  Fact 0.7: The abolition of federal parole did not apply to transfer treaty inmates.
  Fact 0.8: The abolition of federal parole did not apply to persons who violated military law who are in federal civilian prisons.
  Fact 0.9: The abolition of federal parole did not apply to persons who are defendants in state cases and who are under the U.S. Marshals Service Witness Protection Program.

Retrieved items:
<item 1>
Sentencing Reform Act
Sentencing Reform Act - Wikipedia Sentencing Reform Act Jump to : navigation , search The Sentencing Reform Act , part of the Comprehensive Crime Control Act of
1984 , was a U.S. federal statute intended to increase consistency in United States federal sentencing . It established the United States Sentencing Commission
. It also abolished federal parole , except for persons convicted under federal law before November 1 , 1987 , persons convicted under District of Columbia law
, `` transfer treaty '' inmates , persons who violated military law who are in federal civilian prisons , and persons who are defendants in state cases and who
are under the U.S. Marshals Service Witness Protection Program . The act was passed by large majorities in both houses of Congress . References ( edit ) Jump up
^ `` An Overview of the United States Sentencing Commission '' ( PDF ) . United States Sentencing Commission . United States Sentencing Commission . Retrieved 9
August 2011 . Jump up ^ Barry L. Johnson . `` Sentencing Reform Act ( 1984 ) '' . eNotes . Retrieved 9 August 2011 .
</item 1>

<item 2>
United States Bill of Rights
The most frequently litigated clause of the amendment is the last , which forbids cruel and unusual punishment . This clause was only occasionally applied by
the Supreme Court prior to the 1970s , generally in cases dealing with means of execution . In Furman v. Georgia ( 1972 ) , some members of the Court found
capital punishment itself in violation of the amendment , arguing that the clause could reflect `` evolving standards of decency '' as public opinion changed ;
others found certain practices in capital trials to be unacceptably arbitrary , resulting in a majority decision that effectively halted executions in the
United States for several years . Executions resumed following Gregg v. Georgia ( 1976 ) , which found capital punishment to be constitutional if the jury was
directed by concrete sentencing guidelines . The Court has also found that some poor prison conditions constitute cruel and unusual punishment , as in Estelle
v. Gamble ( 1976 ) and Brown v. Plata ( 2011 ) . Ninth Amendment Main article : Ninth Amendment to the United States Constitution The enumeration in the
Constitution , of certain rights , shall not be construed to deny or disparage others retained by the people .
</item 2>

<item 3>
National Flood Insurance Program
The CBRA enacted a set of maps depicting the John H. Chafee Coastal Barrier Resources System ( CBRS ) in which federal flood insurance is unavailable for new or
significantly improved structures . The National Flood Insurance Reform Act of 1994 codified the Community Rating System ( an incentive program that encourages
communities to exceed the minimal federal requirements for development within floodplains ) within the NFIP . The program was further amended by the Flood
Insurance Reform Act of 2004 , with the goal of reducing `` losses to properties for which repetitive flood insurance claim payments have been made . '' The
Biggert -- Waters Flood Insurance Reform Act of 2012 ( Biggert - Waters ) modified the NFIP . At the conclusion of 2011 , as Congress passed Biggert - Waters ,
the NFIP cumulative debt was over $17 billion . A core principle of Biggert - Waters was to change the NFIP premiums to match actuarial risk - based premiums
that better reflected the expected losses and real risk of flooding . These changes included removing discounts to many policies which were being sold below
actual actuarial risk targets and eliminating `` grandfathering '' of older rates .
</item 3>

<item 4>
Capital punishment in Australia
During the 19th century , these crimes saw about 80 people hanged each year throughout Australia . Before and after federation , each state made its own
criminal laws and punishments . For a full list see : Main article : List of people legally executed in Australia Commonwealth ( edit ) In 1973 the Death
Penalty Abolition Act 1973 of the Commonwealth abolished the death penalty for federal offences . It provided in Section 3 that the Act applied to any offence
against a law of the Commonwealth , the Territories or under an Imperial Act , and in s . 4 that `` ( a ) person is not liable to the punishment of death for
any offence '' . No executions were carried out under the bridge of the federal government and the passage of the Death Penalty Abolition Act 1973 saw the death
penalty replaced with life imprisonment as their maximum punishment . Since the Commonwealth effects of utilising this Act no more individuals have been exposed
to the death penalty and it is now replaced with life imprisonment . On 11 March 2010 Federal Parliament passed laws that prevent the death penalty from being
reintroduced by any state or territory in Australia .
</item 4>

<item 5>
United States Bill of Rights
194 . ^ Jump up to : Krantz , Sheldon ( 1986 ) . `` Cruel and Unusual Punishment '' . Encyclopedia of the American Constitution . -- via HighBeam Research (
subscription required ) . Retrieved July 16 , 2013 . Jump up ^ `` U * X * L Encyclopedia of U.S. History '' . UXL Encyclopedia of American History . January 1 ,
2009 . Retrieved July 15 , 2013 . ^ Jump up to : Weisberg , Robert ( 1986 ) . `` Capital Punishment '' . Encyclopedia of the American Constitution . -- via
HighBeam Research ( subscription required ) . Retrieved July 16 , 2013 . Jump up ^ `` The Constitution of the United States of America : Analysis and
Interpretation , Centennial Edition , Interim Edition : Alalysis of Cases Decided by the Supreme Court of the United States to June 26 , 2013 '' ( PDF ) .
Washington , DC : U.S. Government Printing Office . 2013 . pp. 1738 -- 39 .
</item 5>

<item 6>
Sentencing Reform Act
eNotes . Retrieved 9 August 2011 . Jump up ^ `` FREQUENTLY ASKED QUESTIONS ABOUT THE LACK OF PAROLE FOR FEDERAL PRISONERS '' ( Archive ) . Families Against
Mandatory Minimums . p. 2 / 4 . Retrieved on December 13 , 2015 . Jump up ^ Stith , Kate ; Koh , Steve Y. ( 1993 ) . `` The Politics of Sentencing Reform : The
Legislative History of the Federal Sentencing Guidelines '' . Faculty Scholarship Series . Yale Law School . Retrieved 9 August 2011 . This United States
federal legislation article is a stub . You can help Wikipedia by expanding it .
</item 6>

<item 7>
Whistleblower Protection Act
Legislative history Introduced in the Senate as S. 20 by Carl Levin ( D -- MI ) on January 25 , 1989 Passed the Senate on March 16 , 1989 ( 97 - 0 , Roll call
vote 24 , via Senate.gov ) Passed the House on March 21 , 1989 ( Agreed voice vote ) Signed into law by President George H.W. Bush on April 10 , 1989 The
Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the
government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take )
retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file
complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority
; or a substantial and specific danger to public health or safety .
</item 7>

<item 8>
Capital punishment in Australia
The last sentence of death in Western Australia ( indeed anywhere in Australia ) was passed in 1984 , but the female killer ( Brenda Hodge ) in question had her
sentence commuted to imprisonment for life , as was customary by this stage . Capital punishment was formally removed from the statutes of the state with the
passage of the Acts Amendment ( Abolition of Capital Punishment ) Act 1984 . South Australia ( edit ) The Adelaide Gaol was the site of forty - four hangings ,
from Joseph Stagg on 18 November 1840 to Glen Sabre Valance , murderer and rapist , on 24 November 1964 and three executions also occurred at Mount Gambier Gaol
. Two Ngarrindjeri men were controversially executed by hanging along the Coorong on 22 August 1840 , after a drumhead court - martial conducted by Police
Commissioner O'Halloran on the orders of Governor George Gawler . The men were found to be guilty of murdering the twenty - five survivors of the shipwreck
Maria a few months before .
</item 8>

<item 9>
Sentencing Reform Act
This United States federal legislation article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Sentencing_Reform_Act&oldid=813388767 '' Categories : United States federal criminal legislation Penal system in the
United States United States sentencing law 1984 in law 1984 in the United States United States federal legislation stubs Hidden categories : Use dmy dates from
August 2011 All stub articles Talk About Wikipedia Add links This page was last edited on 3 December 2017 , at 12 : 21 . About Wikipedia
</item 9>

<item 10>
Capital punishment in Australia
Australian Federal Police . Retrieved 3 December 2017 . Jump up ^ Finlay , Lorraine ( March 2011 ) . `` Exporting the Death Penalty ? Reconciling International
Police Cooperation and the Abolition of the Death Penalty in Australia '' ( pdf ) . Sydney Law Review . 33 ( 1 ) : 95 -- 117 . ISSN 0082 - 0512 . Retrieved 3
December 2017 . Jump up ^ Traditional Aboriginal Law and Punishment Archived 6 March 2009 at the Wayback Machine . -- Part V -- Aboriginal Customary Law and the
Criminal Justice System , Law Reform Commission of Western Australia -- Aboriginal Customary Laws Discussion Paper Jump up ^ `` British Justice '' . Museum
Victoria Australia . Retrieved 14 March 2017 . Jump up ^ The Hanging of Ronald Ryan : 40 Years Later Archived 28 December 2007 at the Wayback Machine ...
Cosmos.bcst.yahoo.com ( 2009 - 08 - 27 ) . Retrieved on 2012 - 09 - 10 . Jump up ^ Biography -- Jean Lee -- Australian Dictionary of Biography .
</item 10>
=== END ITEM id=a7300cd6a8e0e822 ===

=== ITEM id=f1177690bdb03830 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins , Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone call , but her phone short circuits . Will 's friends Mike , Dustin , and Lucas search for Will in the woods and find Eleven . `` Chapter Two : The Weirdo on Maple Street '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) The boys bring Eleven to Mike 's house , where she sleeps in the basement . The next day , Mike nicknames her `` El '' . She says that `` bad people '' are looking for her and refuses to meet Mike 's parents . Scientists from the laboratory find a substance oozing from the walls of Joyce 's home . At Mike 's home , Eleven recognizes and points out Will in a photo . Dustin and Lucas want to inform Mike 's parents about Eleven , but she uses psychokinesis to stop them . While searching for Will , Mr. Clarke , the boys ' science teacher , discovers a scrap of a hospital gown outside the laboratory grounds . Nancy goes with her friend Barb to a party at her boyfriend Steve 's house . Will 's brother Jonathan investigates the woods where Will went missing . Hearing screaming , he runs to help but finds only Steve , Nancy , and their friends roughhousing around Steve 's swimming pool . He secretly photographs them . Barb , left alone by the poolside , vanishes . Joyce receives another call from Will , hears music from his room , and sees something coming through the wall . `` Chapter Three : Holly , Jolly '' Shawn Levy Jessica Mecklenburg July 15 , 2016 ( 2016 - 07 - 15 ) Barb wakes up in an empty and decaying pool . She screams for Nancy and attempts to climb out but is dragged down by something unseen . Joyce strings Christmas lights around her home to talk to Will , who can turn them on and off . Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab 's security camera footage . Hopper realizes it is fake . He and his deputies research Hawkins Lab , Brenner , and a woman named Terry Ives who claimed her daughter was taken by Brenner . Eleven has a flashback in which Brenner , whom she calls `` Papa , '' has her put in solitary confinement for refusing to telekinetically harm a cat . Nancy worries about Barb , who is missing . Steve and his friends discover Jonathan 's photographs . They destroy them and his camera . Nancy recovers a photo of Barb by the poolside and returns to Steve 's house to search for her . Nancy finds Barb 's car and sees the creature in the woods . Joyce establishes a code with Will using the lights , which he uses to tell her that he is alive but unsafe . He tells Joyce to run as a creature begins to climb through her wall . Will 's apparent body is discovered in the water at a quarry . `` Chapter Four : The Body '' Shawn Levy Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Eleven proves to the boys that Will is still alive by making contact through Mike 's walkie - talkie . Joyce goes to the morgue to view the body and refuses to believe it is Will . The boys give Eleven a makeover so they can sneak her into school . Joyce hears Will in her living room wall . Tearing away the wallpaper , she sees him behind a membrane . At school , Eleven uses Mr. Clarke 's ham radio to channel Will talking to his mother . Joyce demolishes the wall with an axe but reveals only her front porch . Nancy discovers a figure behind Barb in Jonathan 's photo . Jonathan realizes that the figure matches his mother 's description of the creature : a humanoid figure with long arms and no face . Nancy tells the police about her missing friend and subsequently admits to her mother that she slept with Steve . When she tells Steve about the police , she is irritated that he cares only about not getting in trouble with his father . Hopper confronts the state trooper who found Will 's body and beats him until he admits he was ordered to lie . Hopper goes to the morgue and finds that Will 's supposed body is a dummy . He breaks into Hawkins laboratory . 5 5 `` Chapter Five : The Flea and the Acrobat '' The Duffer Brothers Alison Tatlock July 15 , 2016 ( 2016 - 07 - 15 ) Will 's father Lonnie assures Joyce her experiences are hallucinations . Hopper discovers a portal in the basement of the lab but is knocked out by the lab 's guards . He wakes at home and discovers a hidden microphone . Sheriff 's deputies inform Hopper that Barb 's car was found , but Hopper finds the state 's involvement suspicious . The boys determine Will is trapped in an alternate dimension Eleven calls the `` Upside Down , '' and he is being pursued by a monster similar to a Demogorgon . After Will 's funeral , science teacher Mr. Clarke tells the boys that a high - energy spacetime tear could create a passage between dimensions . Using their compasses , they search for a disruption in the Earth 's electromagnetic field . Eleven remembers encountering the creature during a lab experiment , and afraid , she redirects the compasses . Lucas notices and confronts her . Mike defends her , he and Lucas fight , and Eleven telekinetically flings Lucas off Mike . While Dustin and Mike tend to the unconscious Lucas , Eleven disappears . In the woods , Nancy and Jonathan find a wounded deer . The creature drags the deer away , and Nancy follows the trail of blood to a portal to the Upside Down . She crawls through and discovers the creature feasting upon the deer . She snaps a twig and draws the creature 's attention . 6 6 `` Chapter Six : The Monster '' The Duffer Brothers Jessie Nickson - Lopez July 15 , 2016 ( 2016 - 07 - 15 ) Jonathan pulls Nancy back through the portal and saves her from the monster . Later , Steve sees them together through her bedroom window and assumes they are dating . Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store . During a fist fight with Steve , Jonathan inadvertently assaults a police officer and is arrested . Joyce and Hopper track down Terry Ives , who is catatonic and tended by her sister Becky . Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant ; she believed her daughter Jane was kidnapped by Brenner at birth . Joyce and Hopper conclude Eleven is likely Jane . Eleven shoplifts Eggo waffles . She recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement . While searching for Eleven , Mike and Dustin are ambushed by school bullies Troy and James . Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will 's body was discovered . Mike jumps but is levitated to safety by Eleven , who scares away the bullies . Lucas realizes agents from the laboratory are heading to Mike 's house . 7 7 `` Chapter Seven : The Bathtub '' The Duffer Brothers Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Lucas warns Mike via walkie - talkie that government agents are en route . Mike , Dustin , and Eleven flee . Eleven telekinetically flips a van that blocks their path , and the kids escape . Lucas reconciles with Mike and Eleven . Joyce and Hopper are called to the police station because of Jonathan 's assault . Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper . The group contacts Mike and his friends and they rendezvous . They ask Eleven to search for Will and Barb , but her earlier feats have left her too weak to find them . To amplify Eleven 's powers , they break into the middle school and build a sensory - deprivation tank . Eleven finds Barb 's corpse with a slug - like creature crawling out of her mouth . With Joyce 's help , she finds Will alive , hiding in the Upside Down `` Castle Byers , '' his backyard fort . Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards . Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station . In the Upside Down , the monster breaks into the fort where Will is hiding . 8 8 `` Chapter Eight : The Upside Down '' The Duffer Brothers Story by : Paul Dichter Teleplay by : The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) Hopper gives up Eleven 's location so that he and Joyce can enter the Upside Down . They find Will unconscious in the Upside - Down town library . They pull a tendril out of his throat , perform CPR , and revive him . Nancy and Jonathan booby - trap the Byers ' home , then cut their hands to attract the monster with their blood . Steve arrives , intending to apologize to Jonathan about their fight , but the monster attacks . The monster springs the trap but escapes back to the Upside Down . Eleven and the boys hide in the middle school . Mike asks Eleven to a school dance , the Snow Ball , and kisses her . The military storms the school to re-capture Eleven , but she kills them and collapses . As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends . Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from his family .
  Fact 0.0: The table lists, for each episode, its overall number, its number in the season, its title, its director, its writers, and its original release date.
  Fact 0.1: Chapter One: The Vanishing of Will Byers was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.2: Chapter One: The Vanishing of Will Byers was released on July 15, 2016.
  Fact 0.3: On November 6, 1983, in a U.S. Department of Energy laboratory in the town of Hawkins, Indiana, a scientist is attacked by an unseen creature.
  Fact 0.4: While bicycling home from a Dungeons & Dragons session with his friends, 12-year-old Will Byers encounters the creature and vanishes.
  Fact 0.5: The next day, a buzz-cut young girl wearing a hospital gown steals food from a local diner.
  Fact 0.6: The diner's owner, Benny, takes pity on the girl and feeds her before calling social services.
  Fact 0.7: From a tattoo on her arm, Benny learns that the girl's name is Eleven.
  Fact 0.8: A woman posing as a social worker arrives and murders Benny.
  Fact 0.9: Armed men search the diner for Eleven, but she escapes.
  Fact 0.10: Will's mother Joyce believes she hears Will's voice on a distorted phone call, but her phone short circuits.
  Fact 0.11: Will's friends Mike, Dustin, and Lucas search for Will in the woods and find Eleven.
  Fact 0.12: Chapter Two: The Weirdo on Maple Street was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.13: Chapter Two: The Weirdo on Maple Street was released on July 15, 2016.
  Fact 0.14: The boys bring Eleven to Mike's house, where she sleeps in the basement.
  Fact 0.15: The next day, Mike nicknames her "El".
  Fact 0.16: Eleven says that "bad people" are looking for her and refuses to meet Mike's parents.
  Fact 0.17: Scientists from the laboratory find a substance oozing from the walls of Joyce's home.
  Fact 0.18: At Mike's home, Eleven recognizes and points out Will in a photo.
  Fact 0.19: Dustin and Lucas want to inform Mike's parents about Eleven, but she uses psychokinesis to stop them.
  Fact 0.20: While searching for Will, Mr. Clarke, the boys' science teacher, discovers a scrap of a hospital gown outside the laboratory grounds.
  Fact 0.21: Nancy goes with her friend Barb to a party at her boyfriend Steve's house.
  Fact 0.22: Will's brother Jonathan investigates the woods where Will went missing.
  Fact 0.23: Hearing screaming, Jonathan runs to help but finds only Steve, Nancy, and their friends roughhousing around Steve's swimming pool.
  Fact 0.24: Jonathan secretly photographs Steve, Nancy, and their friends.
  Fact 0.25: Barb, left alone by the poolside, vanishes.
  Fact 0.26: Joyce receives another call from Will, hears music from his room, and sees something coming through the wall.
  Fact 0.27: Chapter Three: Holly, Jolly was directed by Shawn Levy and written by Jessica Mecklenburg.
  Fact 0.28: Chapter Three: Holly, Jolly was released on July 15, 2016.
  Fact 0.29: Barb wakes up in an empty and decaying pool.
  Fact 0.30: Barb screams for Nancy and attempts to climb out but is dragged down by something unseen.
  Fact 0.31: Joyce strings Christmas lights around her home to talk to Will, who can turn them on and off.
  Fact 0.32: Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab's security camera footage.
  Fact 0.33: Hopper realizes the footage is fake.
  Fact 0.34: Hopper and his deputies research Hawkins Lab, Brenner, and a woman named Terry Ives, who claimed her daughter was taken by Brenner.
  Fact 0.35: Eleven has a flashback in which Brenner, whom she calls "Papa," has her put in solitary confinement for refusing to telekinetically harm a cat.
  Fact 0.36: Nancy worries about Barb, who is missing.
  Fact 0.37: Steve and his friends discover Jonathan's photographs.
  Fact 0.38: Steve and his friends destroy Jonathan's photographs and his camera.
  Fact 0.39: Nancy recovers a photo of Barb by the poolside and returns to Steve's house to search for her.
  Fact 0.40: Nancy finds Barb's car and sees the creature in the woods.
  Fact 0.41: Joyce establishes a code with Will using the lights, which he uses to tell her that he is alive but unsafe.
  Fact 0.42: Will tells Joyce to run as a creature begins to climb through her wall.
  Fact 0.43: Will's apparent body is discovered in the water at a quarry.
  Fact 0.44: Chapter Four: The Body was directed by Shawn Levy and written by Justin Doble.
  Fact 0.45: Chapter Four: The Body was released on July 15, 2016.
  Fact 0.46: Eleven proves to the boys that Will is still alive by making contact through Mike's walkie-talkie.
  Fact 0.47: Joyce goes to the morgue to view the body and refuses to believe it is Will.
  Fact 0.48: The boys give Eleven a makeover so they can sneak her into school.
  Fact 0.49: Joyce hears Will in her living room wall.
  Fact 0.50: Tearing away the wallpaper, Joyce sees Will behind a membrane.
  Fact 0.51: At school, Eleven uses Mr. Clarke's ham radio to channel Will talking to his mother.
  Fact 0.52: Joyce demolishes the wall with an axe but reveals only her front porch.
  Fact 0.53: Nancy discovers a figure behind Barb in Jonathan's photo.
  Fact 0.54: Jonathan realizes the figure matches his mother's description of the creature: a humanoid figure with long arms and no face.
  Fact 0.55: Nancy tells the police about her missing friend.
  Fact 0.56: Nancy subsequently admits to her mother that she slept with Steve.
  Fact 0.57: When Nancy tells Steve about the police, she is irritated that he cares only about not getting in trouble with his father.
  Fact 0.58: Hopper confronts the state trooper who found Will's body and beats him until he admits he was ordered to lie.
  Fact 0.59: Hopper goes to the morgue and finds that Will's supposed body is a dummy.
  Fact 0.60: Hopper breaks into Hawkins laboratory.
  Fact 0.61: Chapter Five: The Flea and the Acrobat was directed by The Duffer Brothers and written by Alison Tatlock.
  Fact 0.62: Chapter Five: The Flea and the Acrobat was released on July 15, 2016.
  Fact 0.63: Will's father Lonnie assures Joyce her experiences are hallucinations.
  Fact 0.64: Hopper discovers a portal in the basement of the lab but is knocked out by the lab's guards.
  Fact 0.65: Hopper wakes at home and discovers a hidden microphone.
  Fact 0.66: Sheriff's deputies inform Hopper that Barb's car was found.
  Fact 0.67: Hopper finds the state's involvement suspicious.
  Fact 0.68: The boys determine Will is trapped in an alternate dimension that Eleven calls the "Upside Down."
  Fact 0.69: The boys determine Will is being pursued by a monster similar to a Demogorgon.
  Fact 0.70: After Will's funeral, science teacher Mr. Clarke tells the boys that a high-energy spacetime tear could create a passage between dimensions.
  Fact 0.71: Using their compasses, the boys search for a disruption in the Earth's electromagnetic field.
  Fact 0.72: Eleven remembers encountering the creature during a lab experiment.
  Fact 0.73: Afraid, Eleven redirects the compasses.
  Fact 0.74: Lucas notices and confronts Eleven.
  Fact 0.75: Mike defends Eleven, and he and Lucas fight.
  Fact 0.76: Eleven telekinetically flings Lucas off Mike.
  Fact 0.77: While Dustin and Mike tend to the unconscious Lucas, Eleven disappears.
  Fact 0.78: In the woods, Nancy and Jonathan find a wounded deer.
  Fact 0.79: The creature drags the deer away, and Nancy follows the trail of blood to a portal to the Upside Down.
  Fact 0.80: Nancy crawls through the portal and discovers the creature feasting upon the deer.
  Fact 0.81: Nancy snaps a twig and draws the creature's attention.
  Fact 0.82: Chapter Six: The Monster was directed by The Duffer Brothers and written by Jessie Nickson-Lopez.
  Fact 0.83: Chapter Six: The Monster was released on July 15, 2016.
  Fact 0.84: Jonathan pulls Nancy back through the portal and saves her from the monster.
  Fact 0.85: Later, Steve sees Nancy and Jonathan together through her bedroom window and assumes they are dating.
  Fact 0.86: Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store.
  Fact 0.87: During a fist fight with Steve, Jonathan inadvertently assaults a police officer and is arrested.
  Fact 0.88: Joyce and Hopper track down Terry Ives, who is catatonic and tended by her sister Becky.
  Fact 0.89: Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant.
  Fact 0.90: Becky explains that Terry believed her daughter Jane was kidnapped by Brenner at birth.
  Fact 0.91: Joyce and Hopper conclude Eleven is likely Jane.
  Fact 0.92: Eleven shoplifts Eggo waffles.
  Fact 0.93: Eleven recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement.
  Fact 0.94: While searching for Eleven, Mike and Dustin are ambushed by school bullies Troy and James.
  Fact 0.95: Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will's body was discovered.
  Fact 0.96: Mike jumps but is levitated to safety by Eleven, who scares away the bullies.
  Fact 0.97: Lucas realizes agents from the laboratory are heading to Mike's house.
  Fact 0.98: Chapter Seven: The Bathtub was directed by The Duffer Brothers and written by Justin Doble.
  Fact 0.99: Chapter Seven: The Bathtub was released on July 15, 2016.
  Fact 0.100: Lucas warns Mike via walkie-talkie that government agents are en route.
  Fact 0.101: Mike, Dustin, and Eleven flee.
  Fact 0.102: Eleven telekinetically flips a van that blocks their path, and the kids escape.
  Fact 0.103: Lucas reconciles with Mike and Eleven.
  Fact 0.104: Joyce and Hopper are called to the police station because of Jonathan's assault.
  Fact 0.105: Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper.
  Fact 0.106: The group contacts Mike and his friends, and they rendezvous.
  Fact 0.107: The group asks Eleven to search for Will and Barb, but her earlier feats have left her too weak to find them.
  Fact 0.108: To amplify Eleven's powers, the group breaks into the middle school and builds a sensory-deprivation tank.
  Fact 0.109: Eleven finds Barb's corpse with a slug-like creature crawling out of her mouth.
  Fact 0.110: With Joyce's help, Eleven finds Will alive, hiding in the Upside Down "Castle Byers," his backyard fort.
  Fact 0.111: Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards.
  Fact 0.112: Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station.
  Fact 0.113: In the Upside Down, the monster breaks into the fort where Will is hiding.
  Fact 0.114: Chapter Eight: The Upside Down was directed by The Duffer Brothers, with story by Paul Dichter and teleplay by The Duffer Brothers.
  Fact 0.115: Chapter Eight: The Upside Down was released on July 15, 2016.
  Fact 0.116: Hopper gives up Eleven's location so that he and Joyce can enter the Upside Down.
  Fact 0.117: Hopper and Joyce find Will unconscious in the Upside-Down town library.
  Fact 0.118: Hopper and Joyce pull a tendril out of Will's throat, perform CPR, and revive him.
  Fact 0.119: Nancy and Jonathan booby-trap the Byers' home, then cut their hands to attract the monster with their blood.
  Fact 0.120: Steve arrives, intending to apologize to Jonathan about their fight, but the monster attacks.
  Fact 0.121: The monster springs the trap but escapes back to the Upside Down.
  Fact 0.122: Eleven and the boys hide in the middle school.
  Fact 0.123: Mike asks Eleven to a school dance, the Snow Ball, and kisses her.
  Fact 0.124: The military storms the school to re-capture Eleven, but she kills them and collapses.
  Fact 0.125: As Brenner cradles Eleven, the monster attacks, apparently killing him.
  Fact 0.126: The boys hide Eleven in a classroom.
  Fact 0.127: The monster corners the boys and Eleven, but Eleven pins it against a wall.
  Fact 0.128: Eleven says goodbye to Mike, disintegrates the monster, and vanishes.
  Fact 0.129: Will is hospitalized and reunited with his mother, brother, and friends.
  Fact 0.130: Hopper is reluctantly picked up by a black car.
  Fact 0.131: One month later, Nancy and Steve are back together, and both are friends with Jonathan.
  Fact 0.132: Hopper leaves Eggo waffles and other food in a box in the woods.
  Fact 0.133: Will coughs up a slug-like creature and has a vision of the world as the Upside Down, but he hides this from his family.

Retrieved items:
<item 1>
Stranger Things
Gaten Matarazzo , Noah Schnapp , Randy Havens , and Bill Nye Sean Astin October 27 , 2017 ( 2017 - 10 - 27 ) 6 `` The New Class '' Sadie Sink , Brett Gelman ,
Dacre Montgomery , and Linnea Berthelsen Sean Astin and Paul Reiser October 27 , 2017 ( 2017 - 10 - 27 ) 7 `` Closing the Gate '' Shawn Levy , The Duffer
Brothers , Millie Bobby Brown , Noah Schnapp , and Natalia Dyer David Harbour , Sean Astin , and Eddie Tsang October 27 , 2017 ( 2017 - 10 - 27 ) Mobile video
game Netflix and BonusXP developed a free mobile game tie - in for Stranger Things , released to iOS and Android devices on October 4 , 2017 . The game uses a
retro - pixel style of art , similar to games for the Super Nintendo Entertainment System . The game is loosely based on the Stranger Things story after season
one , with the player starting as Chief of Police Jim Hopper looking for the missing boys .
</item 1>

<item 2>
Stranger Things
Stranger Things - wikipedia Stranger Things Jump to : navigation , search For other uses , see Stranger Things ( disambiguation ) . Stranger Things Genre
Science fiction Horror Supernatural Period drama Created by The Duffer Brothers Starring Winona Ryder David Harbour Finn Wolfhard Millie Bobby Brown Gaten
Matarazzo Caleb McLaughlin Natalia Dyer Charlie Heaton Cara Buono Matthew Modine Noah Schnapp Sadie Sink Joe Keery Dacre Montgomery Sean Astin Paul Reiser
Composer ( s ) Kyle Dixon Michael Stein Country of origin United States Original language ( s ) English No. of seasons No.
</item 2>

<item 3>
Stranger Things
Episodes Season 1 ( 2016 ) No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The
Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins ,
Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers
encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes
pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker
arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone
call , but her phone short circuits .
</item 3>

<item 4>
Stranger Things
For the second season , Schnapp and Keery were promoted to series regulars , along with the additions of Sadie Sink , Dacre Montgomery , Sean Astin and Paul
Reiser . Set in the fictional town of Hawkins , Indiana , in the 1980s , the first season focuses on the investigation into the disappearance of a young boy
amid supernatural events occurring around the town including the appearance of a girl with psychokinetic abilities who helps the missing boy 's friends in their
own search . The second season , titled Stranger Things 2 , is set a year after the first , and deals with attempts of the characters to return to normal and
consequences that linger from the first season . The Duffer Brothers developed the series as a mix of investigative drama alongside supernatural elements with
childlike sensibilities , establishing its time frame in the 1980s and creating a homage to pop culture of that decade . Several themes and directorial aspects
were inspired and aesthetically informed by the works of Steven Spielberg , John Carpenter , and Stephen King , among others , including several films , anime
and video games . The first season was released on Netflix on July 15 , 2016 .
</item 4>

<item 5>
Stranger Things
As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it
against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends
. Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo
waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from
his family . Season 2 ( 2017 ) No . overall No. in season Title Directed by Written by Original release date 9 `` Chapter One : MADMAX '' The Duffer Brothers
The Duffer Brothers October 27 , 2017 ( 2017 - 10 - 27 ) On October 28 , 1984 , Kali and her gang use her psychic powers to escape the police . Her wrist tattoo
marks her as `` 008 . ''
</item 5>

<item 6>
Stranger Things
The first season was released on Netflix on July 15 , 2016 . It received critical acclaim for its characterization , pacing , atmosphere , acting , soundtrack ,
directing , writing , and homages to 1980s genre films . The series has received several industry nominations and awards , including winning the Screen Actors
Guild Award for Outstanding Performance by an Ensemble in a Drama Series in 2016 , and receiving eighteen nominations for the 69th Primetime Emmy Awards ,
including Outstanding Drama Series . On August 31 , 2016 , Netflix renewed the series for a second season of nine episodes , which was released on October 27 ,
2017 . In December 2017 , Netflix ordered a third season . The Duffer Brothers have said that Stranger Things is likely to end after its fourth or fifth season
. Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Main 2.2 Recurring 2.2. 1 Introduced in season one 2.2. 2 Introduced in season two 2.3 Guest 2.3. 1
Introduced in season one 2.3.
</item 6>

<item 7>
Stranger Things
Cast and characters The Duffer Brothers , Shawn Levy and the main cast of season 2 of Stranger Things at the 2017 San Diego Comic - Con International Main
Winona Ryder as Joyce Byers , the mother of Will and Jonathan Byers . She is divorced from Lonnie Byers . David Harbour as Jim Hopper , chief of Hawkins Police
Department . After his young daughter died of cancer , Hopper got divorced and lapsed into alcoholism . Finn Wolfhard as Mike Wheeler , son of Karen and Ted ,
younger brother of Nancy , and one of three friends of Will Byers . He is an intelligent and conscientious student and is committed to his friends . He develops
romantic feelings for Eleven . Millie Bobby Brown as Eleven ( `` El '' ) , a young girl with psychokinetic abilities and a limited vocabulary . Her real name is
Jane , the biological daughter of Terry Ives . After escaping from Hawkins Laboratory , where experiments were being performed on her , she befriends Mike ,
Dustin , and Lucas . She develops romantic feelings for Mike . Gaten Matarazzo as Dustin Henderson , one of Will 's friends .
</item 7>

<item 8>
Stranger Things
Birth. Movies. Death . Retrieved October 12 , 2017 . Jump up ^ Squires , John ( October 12 , 2017 ) . `` '' Stranger Things '' Season 1 Getting VHS - Style Blu
- ray Release at Target `` . Bloody Disgusting . Retrieved October 12 , 2017 . Jump up ^ Holloway , Daniel ( August 25 , 2016 ) . `` ' Stranger Things ' Ratings
: Where Series Ranks Among Netflix 's Most Watched '' . Variety . Retrieved August 26 , 2016 . Jump up ^ Butler , Bethoine ( September 23 , 2016 ) . `` '
Stranger Things ' won over most viewers in just two episodes , according to Netflix '' . Washington Post . Retrieved February 6 , 2017 . Jump up ^ Spangler ,
Todd ( August 18 , 2017 ) . `` Netflix 's ' Marvel 's The Defenders ' Poised for Binge - Viewing Pop , Data Indicates '' . Variety . Archived from the original
on August 20 , 2017 .
</item 8>

<item 9>
Stranger Things
of seasons No. of episodes 17 ( list of episodes ) Production Executive producer ( s ) Karl Gajdusek Cindy Holland Brian Wright Matt Thunell Shawn Levy Dan
Cohen The Duffer Brothers Iain Paterson Shannon Tsang Location ( s ) Jackson , Georgia Cinematography Tim Ives Tod Campbell Running time 42 -- 62 minutes
Production company ( s ) 21 Laps Entertainment Monkey Massacre Distributor Netflix Release Original network Netflix Picture format 4K ( Ultra HD ) High dynamic
range Audio format Dolby Digital 5.1 Original release July 15 , 2016 ( 2016 - 07 - 15 ) -- present External links Website Stranger Things is an American science
fiction - horror web television series created , written , directed and co-executive produced by The Duffer Brothers , as well as co-executive produced by Shawn
Levy and Dan Cohen . The first season , released in July 2016 , stars Winona Ryder , David Harbour , Finn Wolfhard , Millie Bobby Brown , Gaten Matarazzo ,
Caleb McLaughlin , Natalia Dyer , Charlie Heaton , Cara Buono and Matthew Modine , with Noah Schnapp and Joe Keery in recurring roles .
</item 9>

<item 10>
Stranger Things
1 Introduced in season one 2.3. 2 Introduced in season two 3 Episodes 3.1 Season 1 ( 2016 ) 3.2 Season 2 ( 2017 ) 3.3 Season 3 4 Production 4.1 Development 4.2
Writing 4.3 Casting 4.4 Filming 4.5 Visual effects 4.6 Music 4.6. 1 Soundtracks 5 Release 5.1 Home media 6 Reception 6.1 Audience viewership 6.2 Critical
reception 6.2. 1 Season 1 6.2. 2 Season 2 6.3 Commentary 6.4 Awards and nominations 7 Other media 7.1 Beyond Stranger Things 7.2 Mobile video game 7.3
PlayStation VR game 7.4 Others 8 References 9 External links Premise Stranger Things is set in the fictional rural town of Hawkins , Indiana , during the early
1980s . The nearby Hawkins National Laboratory ostensibly performs scientific research for the United States Department of Energy , but secretly does
experiments into the paranormal and supernatural , including those that involve human test subjects .
</item 10>
=== END ITEM id=f1177690bdb03830 ===

=== ITEM id=fa6f8037f89a1fb4 ===
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
Florence Cathedral
Unsourced material may be challenged and removed . ( July 2012 ) ( Learn how and when to remove this template message ) Interior of the cathedral Huge clock
decorated by Paolo Uccello . Dante and the Divine Comedy . Trompe l'oeil of Niccolò da Tolentino . Vasari 's fresco begun in 1568 , and completed by Federico
Zuccari in 1579 . Detail under the dome . The Last Judgement ( detail ) under the dome . Tomb of Antonio d'Orso by Tino da Camaino . The Gothic interior is vast
and gives an empty impression . The relative bareness of the church corresponds with the austerity of religious life , as preached by Girolamo Savonarola . Many
decorations in the church have been lost in the course of time , or have been transferred to the Museum Opera del Duomo , such as the magnificent cantorial
pulpits ( the singing galleries for the choristers ) of Luca della Robbia and Donatello .
</item 1>

<item 2>
Florence Cathedral
Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 )
of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing
The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished
the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom
of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of
different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco .
</item 2>

<item 3>
Florence Cathedral
Vasari had used true fresco , while Zuccari had painted in secco . During the restoration work , which ended in 1995 , the entire pictorial cycle of The Last
Judgment was photographed with specially designed equipment and all the information collected in a catalogue . All the restoration information along with
reconstructed images of the frescos were stored and managed in the Thesaurus Florentinus computer system . Crypt ( edit ) Tomb of Filippo Brunelleschi . The
cathedral underwent difficult excavations between 1965 and 1974 . The archaeological history of this huge area was reconstructed through the work of Dr.
Franklin Toker : remains of Roman houses , an early Christian pavement , ruins of the former cathedral of Santa Reparata and successive enlargements of this
church . Close to the entrance , in the part of the crypt open to the public , is the tomb of Brunelleschi . While its location is prominent , the actual tomb
is simple and humble . That the architect was permitted such a prestigious burial place is proof of the high esteem he was given by the Florentines .
</item 3>

<item 4>
Graph query row:
painter: Federico Zuccari
relationship: CREATED
work: The Last Judgment
</item 4>

<item 5>
Florence Cathedral
70 -- 73 . Jump up ^ King , p. 97 . Jump up ^ Mueller , Tom ( February 10 , 2014 ) . `` Mystery of Florence 's Cathedral Dome May Be Solved '' . National
Geographic Society . Retrieved 31 December 2014 . Jump up ^ NationalGeographic.com 2014 - 02 Il Duomo Tom Mueller Jump up ^ NationalGeographic.com 2014 - 02 Il
Doumo Design Video Jump up ^ King , pp. 105 - 107 . Jump up ^ Gartner , Peter , Filippo Brunelleschi 1377 - 1446 , p 95 . Jump up ^ King , p. 69 . Jump up ^
Paolo Galluzzi , `` Leonard de Vinci , engineer and architect '' , p. 50 Jump up ^ Figures vary . archINFORM gives a 45 m wide tambour , while Santa Maria del
Fiore at Structurae gives a 43 m diameter of the cupola , others as little as 42 m .
</item 5>

<item 6>
Florence Cathedral
This fresco , transferred on canvas in the 19th century , in the same style as the previous one , is painted in a color resembling marble . However , it is more
richly decorated and gives more the impression of movement . Both frescoes portray the condottieri as heroic figures riding triumphantly . Both painters had
problems when applying in painting the new rules of perspective to foreshortening : they used two unifying points , one for the horse and one for the pedestal ,
instead a single unifying point . Busts of Giotto ( by Benedetto da Maiano ) , Brunelleschi ( by Buggiano - 1447 ) , Marsilio Ficino , and Antonio Squarcialupi
( a most famous organist ) . These busts all date from the 15th and the 16th centuries . Above the main door is the colossal clock face with fresco portraits of
four Prophets or Evangelists by Paolo Uccello ( 1443 ) . This one - handed liturgical clock shows the 24 hours of the hora italica ( Italian time ) , a period
of time ending with sunset at 24 hours .
</item 6>

<item 7>
Florence Cathedral
Jump up ^ As referenced in `` Cupola di Santa Maria del Fiore : il cantiere di restauro 1980 - 1995 '' , a cura di Cristina Acidini Luchinat e Riccardo Dalla
Negra published by Istituto Poligrafico e Zecca dello Stato ( Roma ) in 1995 ( ISBN 8824039561 ) Jump up ^ Thesaurus Florentinus project page ( in Italian ) ,
Soprintendenza ai Beni Architettonici e Paesaggisitici di Firenze , Ministero dei Beni Culturali References ( edit ) Bartlett , Kenneth R. ( 1992 ) . The
Civilization of the Italian Renaissance . Toronto : D.C. Heath and Company . ISBN 0 - 669 - 20900 - 7 ( Paperback ) . King , Ross ( 2000 ) . Brunelleschi 's
Dome : How a Renaissance Genius Reinvented Architecture . London : Chatto & Windus . Jepson , Tim ( 2001 ) . The National Geographic Traveler , Florence &
Tuscany . National Geographic Society .
</item 7>

<item 8>
Florence Cathedral
Above this shrine is the painting Last Supper by the lesser - known Giovanni Balducci . There was also a glass - paste mosaic panel The Bust of Saint Zanobius
by the 16th - century miniaturist Monte di Giovanni , but it is now on display in the Museum Opera del Duomo . Many decorations date from the 16th - century
patronage of the Grand Dukes , such as the pavement in colored marble , attributed to Baccio d'Agnolo and Francesco da Sangallo ( 1520 -- 26 ) . Some pieces of
marble from the façade were used , topside down , in the flooring ( as was shown by the restoration of the floor after the 1966 flooding ) . It was suggested
that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the
circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end
to this project , and the walls of the dome were whitewashed .
</item 8>

<item 9>
Graph query row:
painter: Giorgio Vasari
relationship: CREATED
work: The Last Judgment
</item 9>

<item 10>
Florence Cathedral
It was the first ' octagonal ' dome in history to be built without a temporary wooden supporting frame . It was one of the most impressive projects of the
Renaissance . During the consecration in 1436 , Guillaume Dufay 's motet Nuper rosarum flores was performed . The structure of this motet was strongly
influenced by the structure of the dome . The Duomo and Baptistery of St. John from Piazza del Duomo The decoration of the exterior of the cathedral , begun in
the 14th century , was not completed until 1887 , when the polychrome marble façade was completed with the design of Emilio De Fabris . The floor of the church
was relaid in marble tiles in the 16th century . The exterior walls are faced in alternate vertical and horizontal bands of polychrome marble from Carrara (
white ) , Prato ( green ) , Siena ( red ) , Lavenza and a few other places . These marble bands had to repeat the already existing bands on the walls of the
earlier adjacent baptistery the Battistero di San Giovanni and Giotto 's Bell Tower .
</item 10>
=== END ITEM id=fa6f8037f89a1fb4 ===

=== ITEM id=737c4691d0c3e006 ===
Gold sentences and facts:
Sentence 0: At the federal level in the United States , legislation ( i.e. , `` statutes '' or `` statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either signed into law by the President or passed by Congress after a presidential veto .
  Fact 0.0: At the federal level in the United States, legislation consists exclusively of Acts.
  Fact 0.1: At the federal level in the United States, legislation is also called statutes or statutory law.
  Fact 0.2: Federal legislation in the United States consists exclusively of Acts passed by the Congress of the United States.
  Fact 0.3: Federal legislation in the United States consists exclusively of Acts passed by the Continental Congress, the predecessor of the Congress of the United States.
  Fact 0.4: Acts passed by the Congress of the United States or the Continental Congress were signed into law by the President.
  Fact 0.5: Acts passed by the Congress of the United States or the Continental Congress were passed by Congress after a presidential veto.

Retrieved items:
<item 1>
List of United States federal legislation
Template : US tax acts , and Category : United States federal taxation legislation Transportation : United States Department of Transportation # Related
legislation Veterans Affairs : United States Department of Veterans Affairs # Related legislation Water Resources , Navigation , Environmental Regulation :
United States Army Corps of Engineers # Public Laws affecting the Corps of Engineers ( partial ) Sources ( edit ) Acts listed by popular name , via Cornell
University Statutes at Large Volumes 1 through 18 , 1789 -- 1875 , via Library of Congress Volumes 19 through 64 , 1875 - 1950 , via Library of Congress Volumes
65 through 125 , 1951 - 2011 , via Government Publishing Office Public laws 93rd Congress through current Congress , via Congress.gov U.S. Code U.S. Code , via
Law Revision Counsel of the U.S. House of Representatives U.S. Code , via Cornell University U.S. Code , via the U.S. Government Publishing Office U.S. Code ,
via FindLaw.com Brian K. Landsberg ( ed ) , Major Acts of Congress .
</item 1>

<item 2>
State legislature (United States)
Constitution . Under Article II , state legislatures choose the manner of appointing the state 's presidential electors . Formerly , state legislatures
appointed the U.S. Senators from their respective states until the ratification of the 17th Amendment in 1913 required the direct election of Senators by the
state 's voters . Lawmaking process ( edit ) Generally , the legislative bodies and their committees use either Mason 's Manual of Legislative Procedure or an
amended form thereof . During official meetings , a professional parliamentarian is available to ensure that legislation and accompanying discussion proceed as
orderly as possible without bias . Beginnings ( edit ) The lawmaking process begins with the introduction of a bill in either the House of Representatives or
the Senate . Bills may be introduced in either house , sometimes with the exception of bills increasing or decreasing revenue , which must originate in the
House of Representatives . The order of business in each house provides a proper time for the introduction of bills . Bills are usually assigned consecutive
numbers , given in the order of their introduction , to facilitate identification . Usually a bill can not become enacted until it has been read on a certain
number of days in each house .
</item 2>

<item 3>
List of United States federal legislation
1789 - 1901 1901 to 2001 : 57th through 106th Congresses ( edit ) Main article : List of United States federal legislation , 1901 - 2001 2001 to present : 107th
and subsequent Congresses ( edit ) Main article : List of United States federal legislation , 2001 - present See also ( edit ) Authorization bill Appropriations
bill ( United States ) List of sources of law in the United States List of Uniform Acts ( United States ) Lists by subject Agriculture : United States
Department of Agriculture # Related legislation Civil Rights : Civil Rights Act ( disambiguation ) Defense : United States Department of Defense # Related
legislation Drugs : Office of National Drug Control Policy # Legislation and executive orders Energy : United States Department of Energy # Related legislation
and Energy law # Federal laws Health and Human Services : United States Department of Health and Human Services # Related legislation Judiciary : Judiciary Act
( disambiguation ) Labor : United States Department of Labor # Related legislation Slavery : Slave Trade Acts Social Security : List of Social Security
legislation ( United States ) Taxation : List of tariffs , Template : US tax acts ,
</item 3>

<item 4>
List of United States federal legislation
List of United States federal legislation - wikipedia List of United States federal legislation Jump to : navigation , search This article includes a list of
references , but its sources remain unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise
citations. ( April 2016 ) ( Learn how and when to remove this template message ) See also : Category : United States federal legislation This is a chronological
, but still incomplete , list of United States federal legislation . Congress has enacted approximately 200 -- 600 statutes during each of its 115 biennial
terms , so that more than 20,000 statutes have been enacted since 1789 . At the federal level in the United States , legislation ( i.e. , `` statutes '' or ``
statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either
signed into law by the President or passed by Congress after a presidential veto . Legislation is not the only source of regulations with the force of law .
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power .
</item 4>

<item 5>
State legislature (United States)
Conference committees ( edit ) A conference committee is often empaneled to discuss the points of difference between the two houses ' versions of the same bill
, and tries to reach an agreement between them so that the identical bill can be passed by both houses . If an agreement is reached and if both houses adopt the
conference committee 's report , the bill is passed . If either house refuses to adopt the report of the conference committee , a motion may be made for further
conference . If a conference committee is unable to reach an agreement , it may be discharged , and a new conference committee may be appointed . Some highly
controversial bills may be referred to several different conference committees . If an agreement is never reached in conference prior to the end of the
legislative session , the bill is lost . When a bill has passed both houses in identical form , it is then ready for transmittal to the Governor . Presentation
to the governor ( edit ) Once a bill reaches the governor , he or she may sign it , which completes its enactment into law . From this point , the bill becomes
an act , and remains the law of the state unless repealed by legislative action , or overturned by a court decision .
</item 5>

<item 6>
List of United States federal legislation
( edit ) Acts of Congress are published in the United States Statutes at Large . Volumes 1 through 18 , which have all the statutes passed from 1789 to 1875 ,
are available on - line at the Library of Congress , here . In the list below , statutes are listed by X Stat . Y , where X is the volume of the Statutes at
Large and Y is the page number , as well as either the chapter or Public Law number . See examples below . Sessions ( Sess. . ) and Chapters ( ch. . ) ( edit )
Each Congress has two to four sessions . Under the numbering system used from 1789 until 1957 , the Acts in each session are numbered sequentially as Chapters .
This numbering included both laws applicable to the general public and laws relating to specific individuals , e.g. , to grant pensions to disabled veterans .
Public laws ( edit ) Since 1957 , Acts of Congress related to the general public have been designated in the form : Public Law X-Y where X is the number of the
ordinal Congress and Y is the number of the chronological order of the public Act in that Congress .
</item 6>

<item 7>
List of United States federal legislation
This numbering is continuous across sessions of Congress and does not include acts relating to individuals , which are analogously given Private Law X-Y
designations . ( From 1901 to 1957 , the Statutes at Large listed both a chapter and a Public / Private number for each act ) See examples below. There are many
laws . Examples ( edit ) The Militia Act of 1862 of July 17 , 1862 , Sess. 2 , ch. 201 , was the 201st Act of the second session of the 37th Congress . The
National Banking Act of February 25 , 1863 , Sess. 3 , ch. 58 , was the 58th Act of the third session of the 37th Congress . The Global Anti-Semitism Review Act
of 2004 of October 16 , 2004 , Pub. L. 108 -- 332 , 118 Stat. 1282 , was the 332nd Act of Congress ( statute ) passed in the 108th Congress . It can be found in
volume 118 of the U.S. Statutes at Large , starting at page 1282 .
</item 7>

<item 8>
List of United States federal legislation
MacMillan Reference Books ( December 2003 ) ISBN 0 - 02 - 865749 - 7 References ( edit ) Jump up ^ See , e.g. , Act of Aug. 11 , 1790 , ch. 44 , 6 Stat. 3 Jump
up ^ About Bills , Resolutions , and Laws United States Congress House of Representatives Senate ( 114th ← 115th → 116th ) Lists of congressional lists Members
and leaders Members Current By length of service Freshmen Youngest members Resident Commissioner of Puerto Rico Delegates Senate Current by seniority Dean of
the Senate Former U.S. Senators living Earliest serving Earliest living Expelled or censured Classes Born outside the U.S. House Current by seniority Dean of
the House Former U.S. Representatives living Oldest living Earliest serving Expelled , censured ,
</item 8>

<item 9>
List of United States federal legislation
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power . See also : Executive orders of the President
; regulations of Executive branch departments and administrative agencies ; and the procedural rules of the federal courts . Contents ( hide ) 1 Publication of
the law 1.1 Statutes at Large ( Stat . ) 1.2 Sessions ( Sess . ) and Chapters ( ch . ) 1.3 Public laws 1.4 Examples 2 Congress of the Confederation 3 United
States Congress 3.1 1789 to 1901 : 1st through 56th Congresses 3.2 1901 to 2001 : 57th through 106th Congresses 3.3 2001 to present : 107th and subsequent
Congresses 4 See also 5 Sources 6 References Publication of the law ( edit ) Statutes at Large ( Stat. . ) ( edit ) Acts of Congress are published in the United
States Statutes at Large .
</item 9>

<item 10>
State legislature (United States)
If the majority vote in favor of the bill , it is recorded as passed . Transmission to second house ( edit ) A bill that is passed in one house is transmitted ,
along with a formal message , to the other house . If the bill is not reported from committee or is not considered by the full house , the bill is defeated .
The house of origin , upon return of its amended bill , may take any one of several courses of action . It may concur in the amendment by the adoption of a
motion to that effect ; then the bill , having been passed by both houses in identical form , is ready for enrollment . Another possibility is that the house of
origin may adopt a motion to non-concur in the amendment , at which point the bill dies . Finally , the house of origin may refuse to accept the amendment but
request that a conference committee be appointed . The other house usually agrees to the request , and the presiding officer of each house appoints members to
the conference committee .
</item 10>
=== END ITEM id=737c4691d0c3e006 ===
