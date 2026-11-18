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

=== ITEM id=cfbb053e83d101dd ===
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
May 26 , 2016 . Retrieved May 27 , 2016 . Jump up ^ `` Final 2016 - 17 TV Rankings : ' Sunday Night Football ' Winning Streak Continues '' . Deadline Hollywood
. May 26 , 2017 . Retrieved May 26 , 2017 . Jump up ^ Ausiello , Michael ( June 26 , 2017 ) . `` Scandal Farewell Update : Final Season Episode Count Revealed
'' . TVLine . Retrieved July 28 , 2017 . Jump up ^ Gelman , Vlada ( January 8 , 2018 ) . `` Quantico Season 3 and Shondaland 's For the People Get ABC Premiere
Dates '' . TVLine . Retrieved January 8 , 2018 . Jump up ^ Bibel , Sara ( April 6 , 2012 ) . `` Thursday Final Ratings : ' Big Bang Theory ' , ' American Idol '
, ' Person of Interest ' , ' Missing ' , ' Up All Night ' Adjusted Up ; ' Scandal ' Adjusted Down '' . TV by the Numbers .
</item 1>

<item 2>
List of Scandal episodes
2012 ( 2012 - 05 - 17 ) 107 7.33 Season 2 ( 2012 -- 13 ) ( edit ) Main article : Scandal ( season 2 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 8 `` White Hat 's Off '' Tom Verica Jenna Bans September 27 , 2012 ( 2012 - 09 - 27 ) 201 6.74 9 `` The
Other Woman '' Stephen Cragg Heather Mitchell October 4 , 2012 ( 2012 - 10 - 04 ) 202 6.56 10 `` Hunting Season '' Ron Underwood Matt Byrne October 18 , 2012 (
2012 - 10 - 18 ) 203 6.17 11 `` Beltway Unbuckled '' Mark Tinker Mark Fish October 25 , 2012 ( 2012 - 10 - 25 ) 204 6.11 12 5 `` All Roads Lead to Fitz '' Steve
Robin Raamla Mohamed November 8 ,
</item 2>

<item 3>
List of Scandal episodes
2017 ( 2017 - 05 - 18 ) 616 5.23 Season 7 ( 2017 -- 18 ) ( edit ) Main article : Scandal ( season 7 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 ``
Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017
- 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.
</item 3>

<item 4>
List of Scandal episodes
List of Scandal episodes - wikipedia List of Scandal episodes Jump to : navigation , search Scandal is an American political thriller television series created
by Shonda Rhimes , which premiered on ABC on April 5 , 2012 . Kerry Washington stars as Olivia Pope , a former White House Communications Director who leaves to
start her own crisis management firm , Pope and Associates , where she works to keep the secrets and protect the public images of the country 's most powerful
and elite . On staff at the new firm are fresh - faced lawyer Quinn Perkins ( Katie Lowes ) ; smooth talking litigator Harrison Wright ( Columbus Short ) ;
investigator Abby Whelan ( Darby Stanchfield ) ; and computer hacker with a CIA past Huck ( Guillermo Diaz ) . The show also focuses on President of the United
States Fitzgerald Grant ( Tony Goldwyn ) ; First Lady Mellie Grant ( Bellamy Young ) ; White House Chief of Staff Cyrus Beene ( Jeff Perry ) ; Vice President
and eventual presidential candidate Sally Langston ( Kate Burton ) ; U.S.
</item 4>

<item 5>
List of Scandal episodes
Attorney David Rosen ( Joshua Malina ) ; Head of NSA , Jake Ballard ( Scott Foley ) ; chairwoman Elizabeth North ( Portia de Rossi ) ; activist Marcus Walker (
Cornelius Smith Jr . ) ; and head of B613 Eli Pope ( Joe Morton ) . As of January 18 , 2018 , 114 episodes of Scandal have aired , including one special . On
February 10 , 2017 , the show was renewed for a seventh season consisting of 18 episodes . It was later announced it will be the final season . Contents ( hide
) 1 Series overview 2 Episodes 2.1 Season 1 ( 2012 ) 2.2 Season 2 ( 2012 -- 13 ) 2.3 Season 3 ( 2013 -- 14 ) 2.4 Season 4 ( 2014 -- 15 ) 2.5 Season 5 ( 2015 --
16 ) 2.6 Season 6 ( 2017 ) 2.7 Season 7 ( 2017 -- 18 ) 3 Specials 4 Webisodes 4.
</item 5>

<item 6>
List of Scandal episodes
2014 ( 2014 - 04 - 17 ) 318 10.57 Season 4 ( 2014 -- 15 ) ( edit ) Main article : Scandal ( season 4 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 48 `` Randy , Red , Superfreak and Julia '' Tom Verica Shonda Rhimes September 25 , 2014 ( 2014 - 09 -
25 ) 401 11.96 49 `` The State of the Union '' Allison Liddi - Brown Heather Mitchell October 2 , 2014 ( 2014 - 10 - 02 ) 402 10.34 50 `` Inside the Bubble ''
Randy Zisk Matt Byrne October 9 , 2014 ( 2014 - 10 - 09 ) 403 9.52 51 `` Like Father , Like Daughter '' Paul McCrane Mark Fish October 16 , 2014 ( 2014 - 10 -
16 ) 404A 9.
</item 6>

<item 7>
List of Scandal episodes
2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 )
705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something
Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 -
01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 ,
</item 7>

<item 8>
List of Scandal episodes
Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01
) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD Specials (
edit ) No . Title Narrator Aired between Original air date U.S.
</item 8>

<item 9>
List of Scandal episodes
`` Gladiators Do n't Run '' `` No More Blood '' Season 5 `` Heavy is the Head '' `` Dog - Whistle Politics '' `` Baby , It 's Cold Outside '' Season 6 ``
Transfer of Power '' Season 7 `` Watch Me '' `` Day 101 '' Characters Olivia Pope Abby Whelan Quinn Perkins See also Judy Smith Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_Scandal_episodes&oldid=821580194 '' Categories : Lists of American drama television series episodes Scandal (
TV series ) episodes Hidden categories : Use mdy dates from July 2017 Articles containing potentially dated statements from January 2018 All articles containing
potentially dated statements Episode list using the default LineColor Official website not in Wikidata Talk Contents About Wikipedia Čeština Deutsch Español
Français Nederlands 日本 語 Polski Português Русский 中文 Edit links This page was last edited on 21 January 2018 , at 10 : 18 . About Wikipedia
</item 9>

<item 10>
List of Scandal episodes
2015 ( 2015 - 09 - 24 ) May 12 , 2016 ( 2016 - 05 - 12 ) 16 10.68 6 16 January 26 , 2017 ( 2017 - 01 - 26 ) May 18 , 2017 ( 2017 - 05 - 18 ) 30 8.16 7 18
October 5 , 2017 ( 2017 - 10 - 05 ) April 19 , 2018 ( 2018 - 04 - 19 ) N / A N / A Episodes ( edit ) Season 1 ( 2012 ) ( edit ) Main article : Scandal ( season
1 ) No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) `` Sweet Baby '' Paul McGuigan Shonda Rhimes
April 5 , 2012 ( 2012 - 04 - 05 ) 101 7.33 `` Dirty Little Secrets '' Roxann Dawson Heather Mitchell April 12 ,
</item 10>
=== END ITEM id=cfbb053e83d101dd ===

=== ITEM id=6b8327380f10d556 ===
Gold sentences and facts:
Sentence 0: `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20 Political Songs '' .
  Fact 0.0: Where Have All the Flowers Gone? is a modern folk-style song.
  Fact 0.1: The melody of Where Have All the Flowers Gone? was written by Pete Seeger in 1955.
  Fact 0.2: The first three verses of Where Have All the Flowers Gone? were written by Pete Seeger in 1955.
  Fact 0.3: The melody of Where Have All the Flowers Gone? was published in Sing Out! magazine.
  Fact 0.4: The first three verses of Where Have All the Flowers Gone? were published in Sing Out! magazine.
  Fact 0.5: Additional verses of Where Have All the Flowers Gone? were added in May 1960 by Joe Hickerson.
  Fact 0.6: Joe Hickerson turned Where Have All the Flowers Gone? into a circular song.
  Fact 0.7: The rhetorical 'where?' and meditation on death of Where Have All the Flowers Gone? place the song in the ubi sunt tradition.
  Fact 0.8: In 2010, the New Statesman listed Where Have All the Flowers Gone? as one of the 'Top 20 Political Songs'.

Retrieved items:
<item 1>
Where Have All the Flowers Gone?
Versions ( edit ) The Kingston Trio recorded the song in 1961 . Believing it to be a traditional song , they claimed authorship , although upon notice from
Seeger they had their name removed and credited Seeger and Hickerson . Seeger acknowledged their success with this song . Their single , with `` O Ken Karanga
'' as the A-side and the hit `` Where Have All The Flowers Gone ? '' the B - side , reached # 21 in the 1962 Billboard Hot 100 chart and # 4 on the Easy
Listening chart . Peter , Paul and Mary included the song on their eponymous debut album ( which spent five weeks as the # 1 album in the country ) in 1962 .
Marlene Dietrich performed the song in English , French , and German . The song was first performed in French ( as `` Qui peut dire où vont les fleurs ? '' ) by
Marlene in 1962 at a UNICEF concert . She also recorded the song in English and in German , the latter titled `` Sag ' mir , wo die Blumen sind '' , with lyrics
translated by Max Colpet .
</item 1>

<item 2>
Where Have All the Flowers Gone?
Roy Orbison recorded a version of the song which appears on the album The Connoisseur 's Orbison . Eddy Arnold and The Needmore Creek Singers recorded the song
on October 9 , 1963 , and released it on the Folk Song Book album released in January 1964 . Vera Lynn recorded the song as the eleventh cut on her 1964 album
`` Among My Souvenirs '' . The Brothers Four recorded the song on their 1964 LP `` More Big Folk Hits '' , Columbia Records , CL - 2213 . The Four Seasons
recorded the song on their 1964 Philips album Born to Wander , PHM 200 129 . Lars Lönndahl recorded the song in 1964 with Swedish lyrics Inga blommor finns det
mer , translated in 1962 by Beppe Wolgers . Joan Baez included the German version ( Sagt Mir wo die Blumen sind ) on her 1965 album Farewell Angelina .
</item 2>

<item 3>
Where Have All the Flowers Gone?
Sing Out ! . 53 ( 2 ) : 76 . ^ Jump up to : Joe Hickerson . `` Where Have All the Flowers Gone ? '' . presentation to SEM ( Society for Ethnomusicology ) , 50th
Annual Meeting in Atlanta ( quoted in thread ) . Mudcat.org . Retrieved October 20 , 2009 . ^ Jump up to : Smith , Ian K ( March 25 , 2010 ) . `` Top 20
Political Songs : Where Have All the Flowers Gone '' . New Statesman . Retrieved March 25 , 2010 . Jump up ^ A Folk Legend 's Fertile Ground , Oberlin Alumni
Magazine , summer 2014 Jump up ^ Notes from : Where Have All the Flowers Gone - The Songs of Pete Seeger ^ Jump up to : Dunaway , David King ( 2008 ) . How Can
I Keep From Singing ? The Ballad of Pete Seeger , pp. 228 - 30 . Random House , Inc . ISBN 0 - 345 - 50608 - 1 .
</item 3>

<item 4>
Where Have All the Flowers Gone?
ISBN 0 - 345 - 50608 - 1 . Jump up ^ Pete Seeger - Little Boxes / Where Have All the Flowers Gone . Discogs.com . Jump up ^ Pete Seeger - The Bitter and the
Sweet . Discogs.com . Jump up ^ Where Have All the Flowers Gone ? Second Hand Songs . Jump up ^ The Kingston Trio time line . The single was released on
December 18 , 1961 by the group on Capitol Records as a 45 single , 4671 . Source : Liner notes for The Kingston Trio : The Capital Years ( Capitol Records
CD7243 8 28498 2 7 ) . Jump up ^ Gilliland , John ( 1969 ) . `` Show 18 - Blowin ' in the Wind : Pop discovers folk music . ( Part 1 ) '' ( audio ) . Pop
Chronicles . University of North Texas Libraries . Track 5 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult Contemporary : 1961 - 2001 . Record Research . p.
137 .
</item 4>

<item 5>
Where Have All the Flowers Gone?
Oğuz Tarihmen Ukrainian Де всі квіти , розкажи De vsi kvity , rozkazhy Maria Burmaka Ukrainian Квіти де ? Kvity De ? Yana Zavarzina Grammy Hall of Fame ( edit )
Pete Seeger 's recording of his composition was inducted into the Grammy Hall of Fame , which is a special Grammy award established in 1973 to honor recordings
that are at least 25 years old and that have `` qualitative or historical significance . '' Pete Seeger : Grammy Hall of Fame Awards Year recorded Title Genre
Label Year inducted 1964 `` Where Have All the Flowers Gone ? '' Folk ( single ) Columbia 2002 See also ( edit ) List of anti-war songs References ( edit ) Jump
up ^ Seeger , Pete . `` Where Have All the Flowers Gone '' . Sing Out ! . 11 ( 5 ) . Jump up ^ Hickerson , Joe ( 2009 -- 2010 ) . `` The Songfinder '' . Sing
Out ! . 53 ( 2 ) : 76 .
</item 5>

<item 6>
Where Have All the Flowers Gone?
Seeger created a song which was subsequently published in Sing Out in 1962 . He recorded a version with three verses on The Rainbow Quest album ( Folkways LP FA
2454 ) released in July 1960 . Later , Joe Hickerson added two more verses with a recapitulation of the first in May 1960 in Bloomington , Indiana . In 2010 ,
the New Statesman listed it as one of the `` Top 20 Political Songs '' . The song appeared on the compilation album Pete Seeger 's Greatest Hits ( 1967 )
released by Columbia Records as CS 9416 . Pete Seeger 's recording from the Columbia album The Bitter and the Sweet ( November 1962 ) , CL 1916 , produced by
John H. Hammond was also released as a Columbia Hall of Fame 45 single as 13 - 33088 backed by `` Little Boxes '' in August , 1965 . Versions ( edit ) The
Kingston Trio recorded the song in 1961 .
</item 6>

<item 7>
Where Have All the Flowers Gone?
The 1964 release of the song as a Columbia Records 45 single , 13 - 33088 , by Pete Seeger was inducted into the Grammy Hall of Fame in 2002 in the Folk
category . Contents ( hide ) 1 Composition 2 Versions 3 Grammy Hall of Fame 4 See also 5 References 6 Bibliography Composition ( edit ) Seeger found inspiration
for the song in October 1955 while he was on a plane bound for a concert at Oberlin College , one of the few venues which would hire him during the McCarthy era
. Leafing through his notebook he saw the passage , `` Where are the flowers , the girls have plucked them . Where are the girls , they 've all taken husbands .
Where are the men , they 're all in the army . '' These lines were taken from the traditional Cossack folk song `` Koloda - Duda '' , referenced in the Mikhail
Sholokhov novel And Quiet Flows the Don ( 1934 ) , which Seeger had read `` at least a year or two before '' . Seeger created a song which was subsequently
published in Sing Out in 1962 .
</item 7>

<item 8>
Where Have All the Flowers Gone?
Johnny Rivers had a 1965 top 40 hit in the US with a folk rock version , reaching # 26 on the Billboard Hot 100 and # 9 in Canada . Grady Martin released an
instrumental version in 1965 on his Instrumentally Yours album . Harry Belafonte has made one recording of it at a benefit concert in Stockholm , Sweden , 1966
on the album BEL - 1 . Lester Flatt and Earl Scruggs included the song on their 1968 album Changin ' Times . Walter Jackson recorded a R & B version in Chicago
for Okeh records in 1967 . Jazz guitarist Wes Montgomery recorded an instrumental version in his trademark style on the 1968 LP Road Song In 1969 , Rufus Harley
recorded a jazz instrumental version on his trademark bagpipes , but the track was never commercially released until it was included in his posthumously
released limited edition collection Courage - The Atlantic Recordings in 2006 . The Landsmen released the song as a 45 single on Arvee . Bill Anderson recorded
the song on his eponymous 1971 album .
</item 8>

<item 9>
Where Have All the Flowers Gone?
Version Title Artist Basque Loreak non dira ? Lou Topet , Harkaitz Cano Belarusian Дзе кветкі ўсе ? Bar Akaryna Bengali কোথায় গেল তারা ? ( Kothay Gelo Tara )
Kabir Suman Catalan Què se n'ha fet d'aquelles flors ? Roslyn Smith , La Marta ( Club Super3 ) Chinese 花 兒 怎麼 不見 了 ? Poon Sow Keng ( 潘秀瓊 ) Croatian Kamo je
cvijeće otišlo ? Monia Verardi Czech Řekni , kde ty kytky jsou Judita Čeřovská , Marie Rottrová , Marta Kubišová Danish Where Have All the Flowers Gone Savage
Rose Dutch Zeg me waar de bloemen zijn Jaap Fischer English Where have all the flowers gone ?
</item 9>

<item 10>
Where Have All the Flowers Gone?
Where Have All the Flowers Gone ? - Wikipedia Where Have All the Flowers Gone ? Jump to : navigation , search For the 2008 documentary , see Where Have All the
Flowers Gone ? ( film ) . 1960 release on The Rainbow Quest LP , Folkways , FA 2454 , by Pete Seeger Columbia Hall of Fame 45 rpm single release as 13 - 33088
featuring the November 1962 version . August 1965 . `` Where Have All The Flowers Gone ? '' Song Language English Composer ( s ) Pete Seeger Lyricist ( s ) Pete
Seeger and Joe Hickerson `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete
Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its
rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20
Political Songs '' .
</item 10>
=== END ITEM id=6b8327380f10d556 ===

=== ITEM id=a832247e62c46d3a ===
Gold sentences and facts:
Sentence 0: We admitted we were powerless over alcohol -- that our lives had become unmanageable . Came to believe that a Power greater than ourselves could restore us to sanity . Made a decision to turn our will and our lives over to the care of God as we understood Him . Made a searching and fearless moral inventory of ourselves . Admitted to God , to ourselves , and to another human being the exact nature of our wrongs . Were entirely ready to have God remove all these defects of character . Humbly asked Him to remove our shortcomings . Made a list of all persons we had harmed , and became willing to make amends to them all . Made direct amends to such people wherever possible , except when to do so would injure them or others . Continued to take personal inventory , and when we were wrong , promptly admitted it . Sought through prayer and meditation to improve our conscious contact with God as we understood Him , praying only for knowledge of His will for us and the power to carry that out . Having had a spiritual awakening as the result of these steps , we tried to carry this message to alcoholics , and to practice these principles in all our affairs .
  Fact 0.0: We admitted we were powerless over alcohol.
  Fact 0.1: We admitted our lives had become unmanageable.
  Fact 0.2: We came to believe that a Power greater than ourselves could restore us to sanity.
  Fact 0.3: We made a decision to turn our will and our lives over to the care of God as we understood Him.
  Fact 0.4: We made a searching and fearless moral inventory of ourselves.
  Fact 0.5: We admitted to God, to ourselves, and to another human being the exact nature of our wrongs.
  Fact 0.6: We were entirely ready to have God remove all these defects of character.
  Fact 0.7: We humbly asked God to remove our shortcomings.
  Fact 0.8: We made a list of all persons we had harmed.
  Fact 0.9: We became willing to make amends to all persons we had harmed.
  Fact 0.10: We made direct amends to such people wherever possible, except when to do so would injure them or others.
  Fact 0.11: We continued to take personal inventory.
  Fact 0.12: When we were wrong, we promptly admitted it.
  Fact 0.13: We sought through prayer and meditation to improve our conscious contact with God as we understood Him.
  Fact 0.14: We prayed only for knowledge of God's will for us.
  Fact 0.15: We prayed only for the power to carry out God's will for us.
  Fact 0.16: Having had a spiritual awakening as the result of these steps, we tried to carry this message to alcoholics.
  Fact 0.17: We tried to practice these principles in all our affairs.
Sentence 1: We admitted we were powerless over alcohol -- that our lives had become unmanageable .
  Fact 1.0: We admitted we were powerless over alcohol.
  Fact 1.1: We admitted our lives had become unmanageable.

Retrieved items:
<item 1>
Twelve-step program
As AA chapters were increasing in number during the 1930s and 1940s , the guiding principles were gradually defined as the Twelve Traditions . A singleness of
purpose emerged as Tradition Five : `` Each group has but one primary purpose -- to carry its message to the alcoholic who still suffers '' . Consequently ,
drug addicts who do not suffer from the specifics of alcoholism involved in AA hoping for recovery technically are not welcome in `` closed '' meetings unless
they have a desire to stop drinking alcohol . The principles of AA have been used to form many numbers of other fellowships specifically designed for those
recovering from various pathologies ; each emphasizes recovery from the specific malady which brought the sufferer into the fellowship . Twelve steps ( edit )
For more details on this topic , see List of Twelve Step alternate wordings . The following are the original twelve steps as published by Alcoholics Anonymous :
We admitted we were powerless over alcohol -- that our lives had become unmanageable . Came to believe that a Power greater than ourselves could restore us to
sanity . Made a decision to turn our will and our lives over to the care of God as we understood Him .
</item 1>

<item 2>
Twelve-step program
The mental obsession is described as the cognitive processes that causes the individual to repeat the compulsive behavior after some period of abstinence ,
either knowing that the result will be an inability to stop or operating under the delusion that the result will be different . The description in the First
Step of the life of the alcoholic or addict as `` unmanageable '' refers to the lack of choice that the mind of the addict or alcoholic affords concerning
whether to drink or use again . The illness of the spiritual dimension , or `` spiritual malady , '' is considered in all twelve - step groups to be self -
centeredness . The process of working the steps is intended to replace self - centeredness with a growing moral consciousness and a willingness for self -
sacrifice and unselfish constructive action . In twelve - step groups , this is known as a spiritual awakening or religious experience . This should not be
confused with abreaction , which produces dramatic , but ephemeral , changes . In twelve - step fellowships , `` spiritual awakening '' is believed to develop ,
most frequently , slowly over a period of time . In accordance with the First Step , twelve - step groups emphasize self - admission by members of the problem
they are recovering from .
</item 2>

<item 3>
Twelve-step program
`` Twelve - step recovery and community service '' . Health & Social Work . 28 ( 2 ) : 137 -- 145 . doi : 10.1093 / hsw / 28.2. 137 . CS1 maint : Multiple names
: authors list ( link ) Lile , B. ( 2003 ) . `` Twelve step programs : An update '' . Addictive Disorders & Their Treatment . 2 ( 1 ) : 19 -- 24 . Morgenstern ,
J. , Bux , D. , Labouvie , E. , Blanchard , K.A. , & Morgan , T.J. ( November 2002 ) . `` Examining mechanisms of action in 12 - step treatment : The role of 12
- step cognitions '' . Journal of Studies on Alcohol . 63 ( 6 ) : 665 -- 672 . PMID 12529066 . CS1 maint : Multiple names : authors list ( link ) Sheehan , T. (
2004 ) . `` Twelve Step Facilitation : A Necessary Treatment for Offenders '' .
</item 3>

<item 4>
Twelve-step program
Made a searching and fearless moral inventory of ourselves . Admitted to God , to ourselves , and to another human being the exact nature of our wrongs . Were
entirely ready to have God remove all these defects of character . Humbly asked Him to remove our shortcomings . Made a list of all persons we had harmed , and
became willing to make amends to them all . Made direct amends to such people wherever possible , except when to do so would injure them or others . Continued
to take personal inventory , and when we were wrong , promptly admitted it . Sought through prayer and meditation to improve our conscious contact with God as
we understood Him , praying only for knowledge of His will for us and the power to carry that out . Having had a spiritual awakening as the result of these
steps , we tried to carry this message to alcoholics , and to practice these principles in all our affairs . In some cases , where other twelve - step groups
have adapted the AA steps as guiding principles , step one is uniquely different for each organization , these are sometimes altered to emphasize principles
important to those particular fellowships , or to remove gender - biased language .
</item 4>

<item 5>
Twelve-step program
Retrieved September 28 , 2012 . Jump up ^ Bill W. ( April 1946 ) . `` Our A.A. Experience Has Taught Us That : '' . The A.A. Grapevine . Alcoholics Anonymous .
2 ( 11 ) . ISSN 0362 - 2584 . OCLC 50379271 . ^ Jump up to : Kurtz LF , Chambon A ( 1987 ) . `` Comparison of self - help groups for mental health '' . Health &
Social Work . 12 ( 4 ) : 275 -- 83 . PMID 3679015 . ^ Jump up to : Ronel , Natti ( 2000 ) . `` From Self - Help to Professional Care : An Enhanced Application
of the 12 - Step Program '' . The Journal of Applied Behavioral Science . 36 ( 1 ) : 108 -- 122 . doi : 10.1177 / 0021886300361006 . ISSN 1552 - 6879 . OCLC
1783135 .
</item 5>

<item 6>
Twelve-step program
Twelve - step program - wikipedia Twelve - step program Jump to : navigation , search A twelve - step program is a set of guiding principles outlining a course
of action for recovery from addiction , compulsion , or other behavioral problems . Originally proposed by Alcoholics Anonymous ( AA ) as a method of recovery
from alcoholism , the Twelve Steps were first published in the 1939 book Alcoholics Anonymous : The Story of How More Than One Hundred Men Have Recovered from
Alcoholism . The method was adapted and became the foundation of other twelve - step programs . As summarized by the American Psychological Association , the
process involves the following : admitting that one can not control one 's alcoholism , addiction or compulsion ; recognizing a higher power that can give
strength ; examining past errors with the help of a sponsor ( experienced member ) ; making amends for these errors ; learning to live a new life with a new
code of behavior ; helping others who suffer from the same alcoholism , addictions or compulsions .
</item 6>

<item 7>
Twelve-step program
. Alcoholics Anonymous World Services . ISBN 1 - 893007 - 16 - 2 . OCLC 32014950 . Jump up ^ `` World Service Board of Trustees Bulletin # 13 : Some thoughts
regarding our relationship to Alcoholics Anonymous '' . Narcotics Anonymous World Services . February 23 , 2007 . Archived from the original on October 6 , 2007
. Retrieved October 7 , 2007 . Jump up ^ `` NA History Chronology '' . January 13 , 2006 . Archived from the original on March 3 , 2009 . Retrieved October 7 ,
2007 . 1953 : ... this committee met regularly ... From the beginning ... 12 Steps ... 12 Traditions ... All Steps had ' We ' Jump up ^ `` The 12 Steps of
Recovery '' . Archived from the original on 2012 - 11 - 05 . Retrieved August 28 , 2010 . Jump up ^ `` The Twelve Steps of Marijuana Anonymous '' . Retrieved
September 28 , 2012 . Jump up ^ Bill W.
</item 7>

<item 8>
Twelve-step program
137.4. 409 . PMID 6987904 . Jump up ^ Alcoholics Anonymous ( 1976 - 06 - 01 ) . `` Appendix II . Spiritual Experience '' . Alcoholics Anonymous . Alcoholics
Anonymous World Services . ISBN 0916856593 . OCLC 32014950 . Archived from the original on 2009 - 04 - 27 . Jump up ^ Hayes , Terrell ( February 2000 ) . ``
Stigmatizing Indebtedness : Implications for Labeling Theory '' . Symbolic Interaction. 23 ( 1 ) : 29 -- 46 . doi : 10.1525 / si. 2000.23. 1.29 . ^ Jump up to :
`` Sponsorship Q&A ( pamphlet ) '' ( PDF ) . New York : Alcoholics Anonymous World Services . Jump up ^ `` NYCMA : What is a Sponsor ? '' . New York Crystal
Meth Anonymous Intergroup . Archived from the original on 2016 - 04 - 04 . Retrieved October 8 , 2007 .
</item 8>

<item 9>
Twelve-step program
Contents ( hide ) 1 Overview 2 History 3 Twelve Steps 4 Twelve Traditions 5 Process 6 Sponsorship 7 Effectiveness 8 Criticism 8.1 Confidentiality 8.2 Cultural
identity 9 See also 10 References 11 Further reading 11.1 Scholarly publications Overview ( edit ) Twelve - step methods have been adapted to address a wide
range of alcoholism , substance - abuse and dependency problems . Over 200 self - help organizations -- often known as fellowships -- with a worldwide
membership of millions -- now employ twelve - step principles for recovery . Narcotics Anonymous was formed by addicts who did not relate to the specifics of
alcohol dependency . Demographic preferences related to the addicts ' drug of choice has led to the creation of Cocaine Anonymous , Crystal Meth Anonymous and
Marijuana Anonymous . Behavioral issues such as compulsion for , and / or addiction to , gambling , crime , food , sex , hoarding , debting and work are
addressed in fellowships such as Gamblers Anonymous , Overeaters Anonymous , Sexaholics Anonymous and Debtors Anonymous .
</item 9>

<item 10>
Twelve-step program
ISSN 1552 - 6879 . OCLC 1783135 . Jump up ^ Alcoholics Anonymous ( 2001 - 06 - 01 ) . `` Chapter 2 : There Is a Solution '' . Alcoholics Anonymous ( PDF ) ( 4th
ed . ) . Alcoholics Anonymous World Services . p. 21 . ISBN 1893007162 . OCLC 32014950 . At some stage of his drinking career , he begins to lose all control of
his liquor consumption , once he starts to drink . Jump up ^ Alcoholics Anonymous ( 2001 - 06 - 01 ) . `` Chapter 2 : There Is a Solution '' . Alcoholics
Anonymous ( PDF ) ( 4th ed . ) . Alcoholics Anonymous World Services . p. 21 . ISBN 1893007162 . OCLC 32014950 . These observations would be academic and
pointless if ( he ) never took the first drink , thereby setting the terrible cycle in motion .
</item 10>
=== END ITEM id=a832247e62c46d3a ===

=== ITEM id=80b5f51b1d7399eb ===
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
Graham Hill ( 1962 Dutch Grand Prix ) , Bruce McLaren ( 1962 Monaco Grand Prix ) , Jim Clark ( 1962 Belgian Grand Prix ) , Dan Gurney ( 1962 French Grand Prix )
Riccardo Patrese ( 1982 Monaco Grand Prix ) , John Watson ( 1982 Detroit Grand Prix ) , Nelson Piquet ( 1982 Canadian Grand Prix ) , Didier Pironi ( 1982 Dutch
Grand Prix ) , Niki Lauda ( 1982 British Grand Prix ) , René Arnoux ( 1982 French Grand Prix ) , Patrick Tambay ( 1982 German Grand Prix ) , Elio de Angelis (
1982 Austrian Grand Prix ) , Keke Rosberg ( 1982 Swiss Grand Prix ) Most different race winners in consecutive races ( starting from first race of the season )
7 ( 2012 ) Jenson Button ( 2012 Australian Grand Prix ) , Fernando Alonso ( 2012 Malaysian Grand Prix ) , Nico Rosberg ( 2012 Chinese Grand Prix ) , Sebastian
Vettel ( 2012 Bahrain Grand Prix ) , Pastor Maldonado ( 2012 Spanish Grand Prix ) ,
</item 1>

<item 2>
List of Formula One driver records
3 days Max Verstappen ( 2014 Japanese Grand Prix ) Finishes Most classified finishes 244 Jenson Button Most race finishes 235 Michael Schumacher Most
consecutive classified finishes 41 Nick Heidfeld ( 2007 French Grand Prix -- 2009 Italian Grand Prix ) Most consecutive race finishes 33 Nick Heidfeld ( 2007
Chinese Grand Prix -- 2009 Italian Grand Prix ) Most consecutive finishes from start of career 27 Esteban Ocon ( 2016 Belgian Grand Prix -- 2017 Mexican Grand
Prix ) Most retirements 148 Andrea de Cesaris Most consecutive retirements 22 ( 18 DNF ) Andrea de Cesaris Most retirements in a season 16 ( 14 DNF ) Andrea de
Cesaris ( from 16 races in 1987 ) Longest time between first and last points finishes 7,749 days Michael Schumacher ( 1991 Italian Grand Prix -- 2012 Brazilian
Grand Prix ) Longest time between successive points finishes 2,
</item 2>

<item 3>
List of Formula One driver records
positions Youngest Oldest Most races without Fastest laps Total Percentage In one season Highest percentage in one season Youngest Oldest Most at the same GP
Podium finishes Total Percentage In one season Consecutive Consecutive from first race of season Youngest Oldest Most career starts without a podium Most career
podiums without a win Most career podiums without a World Championship Most races before scoring a podium finish Points Career points Total races finished in
the points Consecutive races finished in the points Highest average points per race started Highest percentage of races finished in the points Most in one
season Youngest Oldest Most points without a win Most without being World Champion World Champions with fewest career points Most races before scoring points
Most races without scoring points Race leaders Every lap , total For at least one lap , total Youngest Most laps Longest distance Most laps without a win Most
consecutive laps in the lead Multiple driver records Pole and win in same race Most wins from pole position in a season Pole , win and fastest lap in same race
Pole , win , fastest lap ,
</item 3>

<item 4>
List of Formula One driver records
Pastor Maldonado ( 2012 Spanish Grand Prix ) , Mark Webber ( 2012 Monaco Grand Prix ) , Lewis Hamilton ( 2012 Canadian Grand Prix ) Fewest race wins in World
Championship winning year Mike Hawthorn ( 1958 ) Keke Rosberg ( 1982 ) World Champion with fewest career wins Mike Hawthorn , Phil Hill Most race wins in one
season without becoming World Champion 10 Lewis Hamilton ( 2016 ) Podiums Highest percentage of podium finishes in one season 100 Michael Schumacher , 17 podium
finishes out of 17 races in 2002 Most podiums for the same team 116 Michael Schumacher ( Ferrari ) Longest time between successive podium finishes 2,870 days
Alexander Wurz ( 1997 British Grand Prix -- 2005 San Marino Grand Prix ) Most consecutive podium finishes ( starting from debut ) 9 Lewis Hamilton ( 2007
Australian Grand Prix -- 2007 British Grand Prix ) Most podium finishes before a victory 15 Jean Alesi , Mika Häkkinen ,
</item 4>

<item 5>
List of Formula One driver records
1953 Indianapolis 500 Tony Brooks 1957 British Grand Prix 9 4th race Bob Sweikert 1955 Indianapolis 500 Jacques Villeneuve 1996 European Grand Prix Source :
Most races before first win ( edit ) Start Driver Race 130th race Mark Webber 2009 German Grand Prix 123rd race Rubens Barrichello 2000 German Grand Prix 119th
race Jarno Trulli 2004 Monaco Grand Prix 113th race Jenson Button 2006 Hungarian Grand Prix 5 111th race Nico Rosberg 2012 Chinese Grand Prix 6 110th race
Giancarlo Fisichella 2003 Brazilian Grand Prix 7 96th race Mika Häkkinen 1997 European Grand Prix 8 95th race Thierry Boutsen 1989 Canadian Grand Prix 9 91st
race Jean Alesi 1995 Canadian Grand Prix 10 82nd race Eddie Irvine 1999 Australian Grand Prix Valtteri Bottas 2017 Russian Grand Prix Sources : Most races
without a win ( edit ) Driver Entries Starts Best result Andrea de Cesaris 214 208 2nd Nick
</item 5>

<item 6>
List of Formula One driver records
-- 2013 215 9 42 6 Felipe Massa 2002 , 2004 -- 2017 269 11 41 7 Riccardo Patrese 1977 -- 1993 256 6 37 8 Jacques Laffite 1974 -- 1986 176 6 32 Jean Alesi 1989
-- 2001 201 10 Juan Pablo Montoya 2001 -- 2006 94 7 30 Source : Most races before scoring a podium finish ( edit ) Entry Driver Race 91st race Martin Brundle
1992 French Grand Prix 73rd race Mika Salo 1999 German Grand Prix 68th race Jenson Button 2004 Malaysian Grand Prix 67th race Johnny Herbert 1995 Spanish Grand
Prix Pedro de la Rosa 2006 Hungarian Grand Prix 6 60th race Gianni Morbidelli 1995 Australian Grand Prix 7 57th race Felipe Massa 2006 European Grand Prix 8
56th race Mark Webber 2005 Monaco Grand Prix 9 55th race Kamui Kobayashi 2012 Japanese Grand Prix Daniel Ricciardo 2014 Spanish Grand Prix Source : Points (
edit ) Throughout the history of the World Championship ,
</item 6>

<item 7>
List of Formula One driver records
Most career points without being World Champion ( edit ) Driver Points Best WDC finish Felipe Massa 1167 2nd in 2008 Mark Webber 1047.5 3rd in 2010 , 2011 and
2013 Daniel Ricciardo 853 3rd in 2014 and 2016 Valtteri Bottas 756 3rd in 2017 5 Rubens Barrichello 658 2nd in 2002 and 2004 6 David Coulthard 535 2nd in 2001 7
Sergio Pérez 467 7th in 2016 and 2017 8 Max Verstappen 439 5th in 2016 9 Nico Hülkenberg 427 9th in 2014 and 2016 10 Gerhard Berger 385 3rd in 1988 and 1994
Source needed World Champions with fewest career points ( edit ) Driver Points World Champion year ( s ) Phil Hill 98 1961 Jochen Rindt 109 1970 Nino Farina
127.33 1950 Mike Hawthorn 127.
</item 7>

<item 8>
List of Formula One driver records
Piquet 1987 Monaco -- 1987 Portuguese Michael Schumacher 2000 -- 2001 2000 Hungarian -- 2001 Brazilian 2004 European -- 2004 Italian Lewis Hamilton 2007 2007
Australian -- 2007 British Nico Rosberg 2015 2015 Australian -- 2015 British 2016 2016 Belgian -- 2016 Abu Dhabi Source : Most consecutive podium finishes from
first race of season ( edit ) Driver Season Consecutive podium finishes Podiums Michael Schumacher 2002 Australian -- Japanese 17 Fernando Alonso 2006 Bahrain
-- Canadian 9 Lewis Hamilton 2007 Australian -- British 2015 Australian -- British Sebastian Vettel 2011 Australian -- British Nico Rosberg 2015 Australian --
British 7 Nico Rosberg 2014 Australian -- Austrian 8 8 Niki Lauda 1976 Brazilian -- Swedish 7 Alain Prost 1988 Brazilian -- French Michael Schumacher 1994
Brazilian -- French Jenson Button 2009 Australian -- Turkish Source needed Youngest Drivers to score a podium finish ( edit ) ( only the first podium finish for
each driver is listed ) Driver Age Place Race Max
</item 8>

<item 9>
List of Formula One driver records
43 days 1959 United States Grand Prix 10 Jacques Laffite 41 years , 319 days 1985 European Grand Prix Source : Podium finishes ( edit ) Total podium finishes (
edit ) Driver Seasons Entries Podiums Percentage Michael Schumacher 1991 -- 2006 , 2010 -- 2012 308 155 7001503200000000000 ♠ 50.32 % Lewis Hamilton 2007 --
2018 211 119 7001564000000000000 ♠ 56.40 % Alain Prost 1980 -- 1991 , 1993 202 106 7001524800000099999 ♠ 52.48 % Sebastian Vettel 2007 -- 2018 202 101
7001500000000000000 ♠ 50.00 % 5 Fernando Alonso 2001 , 2003 -- 2018 296 97 7001327700000000000 ♠ 32.77 % 6 Kimi Räikkönen 2001 -- 2009 ,
</item 9>

<item 10>
List of Formula One driver records
2009 -- 2011 81 80 6th 10 Marcus Ericsson 2014 -- 2018 79 79 8th Source : Most career podiums without a win ( edit ) Driver Entries Starts Podiums Nick Heidfeld
185 183 13 Stefan Johansson 103 79 12 Chris Amon 108 97 11 Romain Grosjean 127 125 10 5 Martin Brundle 165 158 9 Eddie Cheever 143 132 Jean Behra 53 52 8 Luigi
Villoresi 34 31 8 9 Sergio Pérez 140 137 7 10 Andrea de Cesaris 214 208 5 Source : Most career podiums without a World Championship ( edit ) Driver Seasons
Starts Wins Podiums Rubens Barrichello 1993 -- 2011 326 11 68 David Coulthard 1994 -- 2008 247 13 62 Gerhard Berger 1984 -- 1997 210 10 48 Carlos Reutemann 1972
-- 1982 146 12 45 5 Mark Webber 2002
</item 10>
=== END ITEM id=80b5f51b1d7399eb ===

=== ITEM id=5c4055f283fa5d6f ===
Gold sentences and facts:
Sentence 0: The story in the book takes place in Tulsa , Oklahoma , in 1965 , but this is never explicitly stated in the book .
  Fact 0.0: The story in the book takes place in Tulsa, Oklahoma.
  Fact 0.1: The story in the book takes place in 1965.
  Fact 0.2: The book never explicitly states that the story takes place in 1965.

Retrieved items:
<item 1>
The Outsiders (novel)
The story is told in first - person perspective by teenaged protagonist Ponyboy Curtis . The story in the book takes place in Tulsa , Oklahoma , in 1965 , but
this is never explicitly stated in the book . A film adaptation was produced in 1983 , and a little - known short - lived television series appeared in 1990 ,
picking up where the movie left off . A stage adaptation was written by Christopher Sergel and published in 1990 . Contents ( hide ) 1 Plot summary 2 Major
characters 2.1 Greasers 2.2 Socs 2.3 Other characters 3 Controversy 4 See also 5 References 6 Bibliography 7 External links Plot summary Ponyboy Curtis , a
teenaged member of a loose gang of `` greasers '' , is leaving a movie theater when he is jumped by `` Socs '' , the greasers ' rival gang . Several greasers ,
including Ponyboy 's two older brothers -- the paternal Darry and the popular Sodapop -- come to his rescue .
</item 1>

<item 2>
The Outsiders (novel)
After some heated talk , Ponyboy spits at the Socs , prompting them to try to drown him in a nearby fountain , but Johnny stabs Bob , killing him and dispersing
the Socs . Terrified of what to do next , Ponyboy and Johnny rush to find Dally , who gives them money and a loaded gun , directing them to hide in an abandoned
church in Windrixville . During their stay there , Pony cuts and dyes his hair as a disguise , reads Gone with the Wind to Johnny , and , upon viewing a
beautiful sunrise , recites the poem `` Nothing Gold Can Stay '' by Robert Frost . Days later , Dally comes to check on them , revealing that violence between
the greasers and Socs has escalated since Bob 's death into all - out city - wide warfare , with Cherry acting out of guilt as a spy for the greasers . Johnny
decides to turn himself in and Dally agrees to take the boys back home . As they attempt to leave , they notice the church has caught fire and several local
schoolchildren have become trapped inside .
</item 2>

<item 3>
The Outsiders (novel)
The greasers run inside the burning church to save the children , but Ponyboy is rendered unconscious by the fumes . At the hospital he discovers that he and
Dally are not badly injured , but a piece of the church roof fell on Johnny and broke his back . Sodapop and Darry come to the hospital ; Darry breaks down and
cries . Ponyboy then realizes that Darry cares about him , and is only hard on Ponyboy because he loves him and cares about his future . The following morning
the newspapers declare Pony and Johnny heroes , but Johnny will be charged with manslaughter for Bob 's death . Two - Bit tells them that the greaser -- Soc
rivalry is to be settled in a final rumble . Ponyboy and Two - Bit are approached by a Soc named Randy , Bob 's best friend , who expresses remorse for his
involvement in the gang war , lacks confidence about the rumble ending the feud , and says he will not participate . Later , Ponyboy visits Johnny at the
hospital , where he is in critical condition . On their way home , Pony spots Cherry and they talk .
</item 3>

<item 4>
The Outsiders (novel)
Ponyboy returns to school , but his grades drop . Although he is failing English , his teacher , Mr. Syme , says he will pass him if he writes a decent theme .
In the copy of Gone with the Wind that Johnny gave him before dying , Ponyboy finds a note from Johnny describing how he will die proudly after saving the kids
from the fire . Johnny also urges Ponyboy to `` stay gold '' . Ponyboy decides to write his English assignment about the recent events , and begins his essay
with the opening line of the novel : `` When I stepped out into the bright sunlight from the darkness of the movie house , I had only two things on my mind :
Paul Newman and a ride home ... '' Major Characters Greasers Ponyboy Curtis : Narrator and the youngest Curtis brother , 14 years old , who gets good grades and
runs track . Sodapop Curtis : The middle Curtis brother , 16 years old , a high school dropout who works at a gas station . Darrel `` Darry '' Curtis : The
oldest of the Curtis brothers , 20 years old , who has been caring for his brothers since their parents died in a car crash .
</item 4>

<item 5>
The Outsiders (novel)
The Outsiders ( novel ) - wikipedia The Outsiders ( novel ) Jump to : navigation , search The Outsiders First hardcover edition , 1967 Author S.E. Hinton Cover
artist Robert Hunt Country United States Language English Genre Young adult fiction Publisher Viking Press , Dell Publishing Publication date April 24 , 1967
Media type Print ( hardcover , paperback ) , Audiobook Pages 192 ISBN 0 - 670 - 53257 - 6 OCLC 64396432 Followed by That Was Then , This Is Now The Outsiders is
a coming - of - age novel by S.E. Hinton , first published in 1967 by Viking Press . Hinton was 15 when she started writing the novel , but did most of the work
when she was 16 and a junior in high school . Hinton was 18 when the book was published . The book details the conflict between two rival gangs divided by their
socioeconomic status : the working - class `` greasers '' and the upper - class `` Socs '' ( pronounced / ˈsoʊʃɪz / -- short for Socials ) .
</item 5>

<item 6>
The Outsiders (novel)
Jump up ^ Cummings , Siena ( 2017 ) . `` ( www.standard.net/TX/2017/09/24/After-50-years-Outsiders After 50 years , ' Outsiders ' continues to ' stay gold ' ) .
'' Standard - Examiner . Ogden Publishing Corporation . Bibliography Hinton , S.E. ( 1997 ) . The Outsiders . New York : SPEAK . ISBN 9780140385724 . External
links S.E. Hinton 's Website `` ' The Outsiders ' : 40 Years Later '' By Dale Peck The New York Times September 23 , 2007 S.E. Hinton Books The Outsiders Rumble
Fish That Was Then , This Is Now Tex Taming the Star Runner Hawkes Harbor Some of Tim 's Stories Big David , Little David The Puppy Sister Books I Love Best
Yearly : Older Readers Award The Secret Diary of Adrian Mole , Aged 133⁄4 by Sue Townsend ( 1990 ) The Outsiders by S.E. Hinton ( 1991 ) Where 's Wally ?
</item 6>

<item 7>
The Outsiders (novel)
He is the unofficial leader of the gang . Johnny Cade : Ponyboy 's best friend , 16 years old , who is extremely quiet and lives with his alcoholic and abusive
mother and father . Dallas `` Dally '' Winston : The roughest of the greasers , who lived on the streets of New York City for three years . Keith `` Two - Bit
'' Mathews : A wise - cracking kleptomaniac . Steve Randle : Sodapop 's best friend since grade school . Timothy `` Tim '' Shepard : Leader of another greaser
gang : an organized downtown one . Sandy : Sodapop 's girlfriend . Socs Sherri `` Cherry '' Valance : Bob 's girlfriend , attends the same high school as
Ponyboy . Robert `` Bob '' Sheldon : Cherry 's boyfriend who is stabbed and killed by Johnny . Randy Adderson : A friend of Bob 's and Marcia 's boyfriend .
Marcia : Cherry 's best friend and Randy 's girlfriend . David : A member of Bob and Randy 's gang who attempted to drown Ponyboy in the fountain .
</item 7>

<item 8>
The Outsiders (novel)
The next night , Ponyboy and two greaser friends , the hardened Dally and the quiet Johnny , meet Cherry and Marcia , a pair of Soc girls , at a drive - in
movie theater . Cherry spurns Dally 's rude advances , but Ponyboy ends up speaking civilly with Cherry , emotionally connecting with a Soc for the first time
in his life . Afterward , Ponyboy , Johnny , and their wisecracking friend Two - Bit begin to walk Cherry and Marcia home , when they are stopped by Cherry 's
boyfriend Bob , who badly beat up Johnny a few months back . Bob and the greasers exchange taunts , but Cherry prevents a fight by willingly leaving with Bob .
Ponyboy gets home at two in the morning , enraging Darry until he suddenly slaps Ponyboy . Ponyboy runs out the door and meets up with Johnny , expressing his
anger at Darry 's increasing coldness in the wake of his parents ' recent deaths in a car crash . Ponyboy and Johnny wander into a park , where Bob and four
other Socs surround them .
</item 8>

<item 9>
The Outsiders (novel)
On their way home , Pony spots Cherry and they talk . Cherry says she is unwilling to visit Johnny in the hospital because he killed her boyfriend . Pony calls
her a traitor , but after explaining herself , they end on good terms . After escaping the hospital , Dally shows up just in time for the rumble . The greasers
win the brutal fight . Afterward , Pony and Dally hurry back to the hospital to see Johnny , but he dies moments later and a maniacal Dally runs out of the room
. Pony returns home that night feeling confused and disoriented . Dally calls the house to say that he has robbed a store and is running from the police . The
greasers find Dally deliberately pointing an unloaded gun at the police , causing them to shoot and kill him . Overwhelmed , Ponyboy faints and is sick in bed
for several days due to the resulting concussion . When the hearing finally comes , the judge frees Ponyboy from responsibility for Bob 's death and allows Pony
to remain at home with Darry and Soda . Ponyboy returns to school , but his grades drop .
</item 9>

<item 10>
The Outsiders (novel)
Paul Holden : A former friend of Darry 's . Other Characters Jerry Wood : The schoolteacher responsible for the children who were in the seemingly abandoned
church . Mr. Syme : Ponyboy 's English teacher who assigns him a theme to write that becomes the story of the novel itself . Controversy The Outsiders was a
controversial book at the time of its publication ; it is still currently challenged and debated . It was ranked # 38 on the American Library Association 's Top
100 Most Frequently Challenged Books of 1990 -- 1999 . This book has been banned from some schools and libraries because of the portrayal of gang violence ,
underage smoking and drinking , strong language / slang , and family dysfunction . However , in many U.S. schools , the book is part of the English curriculum
at the middle - or high - school level . See also The Outsiders House Museum References Jump up ^ Hinton , S.E. ( 2005 ) ( 1977 ) . `` speaking with S.E. Hinton
'' . The Outsiders . Speak / Penguin Putnam . p. 162 .
</item 10>
=== END ITEM id=5c4055f283fa5d6f ===

=== ITEM id=b1dce74cdee8ea12 ===
Gold sentences and facts:
Sentence 0: Teri Garr as Mrs. Ramsay , Alec 's mother
  Fact 0.0: Teri Garr played the role of Mrs. Ramsay
  Fact 0.1: Mrs. Ramsay is Alec's mother

Retrieved items:
<item 1>
The Black Stallion (film)
The hungry stallion finds himself unable to resist , but visibly struggles with his distrust for humans . Eventually , the hunger wins and he takes Alec 's
offer ; their bond has been sealed and the two are now inseparable . Alec even manages to ride the unbroken horse , after many times falling off the horse . One
day , a fishing ship arrives , rescuing both Alec and the stallion . Back home in Flushing , NY , Alec is given a hero 's welcome . The Black has a temporary
home in Alec 's backyard , but a garbage man not knowing that there is a wild horse in the backyard is chased by the Black , who races off down the street after
being spooked by a passing car . Alec chases after him through every part of town , but loses track of him . The next day , Alec meets Snoe ( and Napoleon ) who
tell him where the Black is . Alec finds the stallion in the barn of Henry Dailey , a retired jockey and racehorse trainer , who apparently spent all night
catching the Black . Henry allows Alec to keep the Black in his barn .
</item 1>

<item 2>
The Black Stallion (film)
After being rescued , they are set on entering a race challenging two champion horses . The film is adapted by Melissa Mathison , Jeanne Rosenberg and William
D. Wittliff . It is directed by Carroll Ballard . The movie stars Kelly Reno , Mickey Rooney , Teri Garr , Hoyt Axton , and the Arabian horse Cass Ole . The
film features music by Carmine Coppola , the father of Hollywood producer Francis Ford Coppola , who was the executive producer of the film . Contents ( hide )
1 Plot 2 Cast 2.1 Horses 3 Awards and honors 3.1 Academy Awards 3.2 Golden Globe Awards 3.3 British Academy Awards 3.4 LA Film Critics Awards 3.5 Others 4
Sequel , prequel and television series 5 See also 6 References 7 External links Plot ( edit ) In 1946 Alec Ramsey is aboard the steamer Drake off the coast of
North Africa , where he sees a wild black stallion being forced into a makeshift stable compartment and heavily restrained by ropes leading to his halter .
</item 2>

<item 3>
The Black Stallion (film)
He falls behind , but the Black wo n't stop . When Alec regains his balance , the Black is well on his way to catch up with his opponents . Alec now encourages
the Black to run as fast as he can , remembering the wild rides on the island , as they catch up . The Black eventually wins by two lengths . Cast ( edit )
Kelly Reno as Alec Ramsay Mickey Rooney as Henry Dailey Teri Garr as Mrs. Ramsay , Alec 's mother Hoyt Axton as Mr. Ramsay , Alec 's father Clarence Muse as
Snoe Michael Higgins as Jim Neville Ed McNamara as Jake Doghmi Larbi as Arab John Burton as Jockey No. 1 John Buchanan as Jockey No.
</item 3>

<item 4>
The Black Stallion (film)
Alec 's father rescues Alec from his attacker but the life jacket is lost and father & son are separated . The black stallion breaks free from his compartment
and then jumps into the sea . Alec himself is swept overboard by a gigantic wave . Once in the water , he swims toward the stallion and managed to grab hold of
the ropes that were previously restraining the stallion . Alec wakes on the shore of a deserted island and starts to explore . He finds the stallion caught in
his restraints with the ropes stuck between the rocks . With his knife , Alec manages to free the stallion once again and the stallion runs away . For a time ,
the two keep their distance . Alec discovers means to survive by catching fish and seaweed . As Alec suddenly faces a cobra eye to eye , the Black comes to the
rescue and kills the snake , only to run off again . By now , Alec decides to try to get closer to the horse and offer him some seaweed . The hungry stallion
finds himself unable to resist , but visibly struggles with his distrust for humans .
</item 4>

<item 5>
Graph query row:
person: Teri Garr
work: The Black Stallion
relationship: RELATED_TO
role: star
context: None
fact: None
</item 5>

<item 6>
The Black Stallion (film)
The Black stallion ( film ) - wikipedia The Black stallion ( film ) Jump to : navigation , search The Black Stallion 1979 poster Directed by Carroll Ballard
Produced by Fred Roos Francis Ford Coppola Tom Sternberg Screenplay by Melissa Mathison Jeanne Rosenberg William D. Wittliff Based on The Black Stallion by
Walter Farley Starring Kelly Reno Mickey Rooney Teri Garr Music by Carmine Coppola Cinematography Caleb Deschanel Edited by Robert Dalva Production company
American Zoetrope Distributed by United Artists Release date October 17 , 1979 ( 1979 - 10 - 17 ) Running time 118 minutes Country United States Language
English Budget $2.7 million Box office $37.8 million The Black Stallion is a 1979 American adventure film based on the 1941 classic children 's novel The Black
Stallion by Walter Farley . But the film starts in 1946 , five years after the book was published . It tells the story of Alec Ramsey , who is shipwrecked on a
deserted island with a wild Arabian stallion whom he befriends .
</item 6>

<item 7>
The Black Stallion (film)
In 2002 , it was selected for preservation in the United States National Film Registry by the Library of Congress as being `` culturally , historically , or
aesthetically significant '' . Also , the film is recognized by American Film Institute in these lists : 2001 : AFI 's 100 Years ... 100 Thrills -- Nominated
2005 : AFI 's 100 Years of Film Scores -- Nominated 2006 : AFI 's 100 Years ... 100 Cheers -- # 64 Sequel , prequel and television series ( edit ) The film was
followed in 1983 by a sequel , The Black Stallion Returns , which also starred Reno . There was also a television series called The Adventures of the Black
Stallion which aired from 1990 to 1993 and starred Mickey Rooney and Richard Ian Cox . In 2003 , a 50 - minute prequel called The Young Black Stallion , was
shot and released for IMAX theaters . See also ( edit ) The Black Stallion books References ( edit ) Jump up ^ `` The Black Stallion , Box Office Information ''
. The Numbers .
</item 7>

<item 8>
The Black Stallion (film)
American Film Institute . Retrieved 2016 - 08 - 20 . External links ( edit ) Official website The Black Stallion on IMDb The Black Stallion at AllMovie The
Black Stallion at Rotten Tomatoes The Black Stallion at Box Office Mojo The Black Stallion at the TCM Movie Database The Black Stallion by Walter Farley Books
The Black Stallion book series Films The Black Stallion ( 1979 ) The Black Stallion Returns ( 1983 ) The Young Black Stallion ( 2003 ) Television The Adventures
of the Black Stallion ( 1990 - 1993 ) Related Cass Ole El Mokhtar Docs Keepin Time Films directed by Carroll Ballard The Black Stallion ( 1979 ) Never Cry Wolf
( 1983 ) Nutcracker : The Motion Picture ( 1986 ) Wind ( 1992 ) Fly Away Home ( 1996 ) Duma ( 2005 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
The Black Stallion (film)
wikipedia.org/w/index.php?title=The_Black_Stallion_(film)&oldid=836268326 '' Categories : 1979 films English - language films American children 's films Films
about horses United States National Film Registry films United Artists films American Zoetrope films Directorial debut films Films shot in Sardinia Films
directed by Carroll Ballard Films that won the Best Sound Editing Academy Award Horse racing films Screenplays by Melissa Mathison American films Films set on
islands Films set in 1946 Hidden categories : Pages using div col with deprecated parameters Talk Contents About Wikipedia Català Dansk Deutsch Español فارسی
Français Bahasa Indonesia Italiano Bahasa Melayu Nederlands 日本 語 Polski Português Русский Svenska 6 more Edit links This page was last edited on 13 April 2018 ,
at 18 : 25 . About Wikipedia
</item 9>

<item 10>
The Black Stallion (film)
1 John Buchanan as Jockey No. 2 Kristen Vigard as Becky Fausto Tozzi as Rescue Captain John Karlsen as Archeologist Leopoldo Trieste as Priest Frank Cousins as
African Chieftain Don Hudson as Zaurog Marne Maitland as Drake Captain Tom Dahlgren as Veterinarian Horses ( edit ) Cass Ole , a champion Arabian stallion , was
featured in most of the movie 's scenes , with Fae Jur , another black Arabian stallion , being his main double . Fae Jur 's main scene is the one where Alec is
trying to gain the trust of the Black on the beach . Two other stunt doubles were used for running , fighting and swimming scenes . El Mokhtar , an Egyptian
Arabian racehorse , was the producers ' first choice to portray the Black , but they were unable to secure his services for the film from his owners , who
declined any offers . He does appear in The Black Stallion Returns , alongside Cass Ole , by which time the studio bought out the syndicate of owners in order
to secure El Mokhtar 's services .
</item 10>
=== END ITEM id=b1dce74cdee8ea12 ===
