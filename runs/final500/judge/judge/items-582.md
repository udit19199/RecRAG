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

This file has 12 items.

=== ITEM id=22bdfbba04efc933 ===
Gold sentences and facts:
Sentence 0: The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
  Fact 0.0: The Nashville Tribute Band is a Mormon music group.
  Fact 0.1: The Nashville Tribute Band was founded by Jason Deere.
  Fact 0.2: The Nashville Tribute Band was founded by Dan Truman.
  Fact 0.3: Dan Truman is the pianist of the popular country group Diamond Rio.
Sentence 1: Diamond Rio is an American country and Christian country music band.
  Fact 1.0: Diamond Rio is an American country music band.
  Fact 1.1: Diamond Rio is a Christian country music band.
Sentence 2:  The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then the Tennessee River Boys.
  Fact 2.0: Diamond Rio was founded in 1982.
  Fact 2.1: Diamond Rio was founded as an attraction for the Opryland USA theme park in Nashville, Tennessee.
  Fact 2.2: Diamond Rio was originally known as the Grizzly River Boys.
  Fact 2.3: Diamond Rio was later known as the Tennessee River Boys.

Retrieved items:
<item 1>
The Studdogs
The Studdogs was a rock band from Orlando, Florida.

 The band's music contains elements of punk, blues rock, garage rock, and noise rock.

 The band was formed in 2000 and stayed active until 2005.

 They released a lot of material, particularly the full-length album, "The Gospel According to the Studdogs" and the 7" vinyl 3-song EP, and gained local and
national notoriety.

 They were featured in several national music magazines, including Magnet and Amplifier.

 Comparisons were made to the Stooges, Dead Boys, and it was said they sounded "like absolute victory by way of slurring, Rolling Stones-y garage rock from
nowhere in particular."

 Their drunken, abrasive live shows often ended with someone bleeding or being thrown out of the venue.

 Their live shows were said to be "what Howling Wolf would sound like through a wall of noise and distortion."
</item 1>

<item 2>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 2>

<item 3>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 3>

<item 4>
Nashville Tribute Band
The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
</item 4>

<item 5>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 5>

<item 6>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 6>

<item 7>
The Pilgrims (band)
The Pilgrims are an alternative rock band from Windsor, Vermont.

 Seven Days contributor Dan Bolles compared their style to alt-punk and indie rock, while John Powell, writing for www.angelica-music.com, has likened them to
rock and roll bands from the 1960s and 1970s.

 They are part of the Windsor, VT music collective What Doth Life, which has released albums by The Pilgrims, Giant Travel Avant Garde, Derek and The Demons,
Carton, and Luke Chrisinger.
</item 7>

<item 8>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 8>

<item 9>
Blackdiamondskye
Blackdiamondskye was a 2010 North American concert tour by American bands Alice in Chains, Mastodon, and Deftones.

 Blackdiamondskye is a portmanteau of the three band's most recent albums: Alice in Chains' "Black Gives Way to Blue", Deftones' "Diamond Eyes", and Mastodon's
"Crack the Skye".

 The tour kicked off on September 16 in Chicago and culminated on October 16 in Paradise, Nevada.

 The tour lasted for 19 shows over the course of one month and featured shows at both outdoor amphitheatres and indoor venues.
</item 9>

<item 10>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 10>
=== END ITEM id=22bdfbba04efc933 ===

=== ITEM id=706d80e2f3b05124 ===
Gold sentences and facts:
Sentence 0:  He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".
  Fact 0.0: The person described has written a column for Fortune magazine for more than twenty years.
  Fact 0.1: The person described spent a decade at Esquire before writing for Fortune.
Sentence 1:  Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
  Fact 1.0: The described publication was founded in 1933.
  Fact 1.1: The described publication flourished during the Great Depression.
  Fact 1.2: The described publication flourished under the guidance of founders Arnold Gingrich, David A. Smart, and Henry L. Jackson.

Retrieved items:
<item 1>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 1>

<item 2>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 2>

<item 3>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 3>

<item 4>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 4>

<item 5>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 5>

<item 6>
This Is Bing Crosby
This Is Bing Crosby was a fifteen-minute five times a week daytime radio program featuring Bing Crosby acting as a disc jockey.

 Minute Maid quick frozen concentrated orange juice was promoted on the shows.
</item 6>

<item 7>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 7>

<item 8>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 8>

<item 9>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 9>

<item 10>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 10>
=== END ITEM id=706d80e2f3b05124 ===

=== ITEM id=695d717be16be632 ===
Gold sentences and facts:
Sentence 0: John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.
  Fact 0.0: John Alexander was born on 5 October 1955.
  Fact 0.1: John Alexander is English.
  Fact 0.2: John Alexander is a former footballer.
  Fact 0.3: John Alexander holds the post of club secretary at Manchester United.
Sentence 1:  Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in 1976.
  Fact 1.0: John Alexander was born in Liverpool.
  Fact 1.1: John Alexander began his football career with a club called Ulysses.
  Fact 1.2: John Alexander was later picked up by Millwall.
  Fact 1.3: John Alexander made his league debut for Millwall in 1976.
Sentence 2:  He is the uncle of Liverpool player Trent Alexander-Arnold.
  Fact 2.0: John Alexander is the uncle of Liverpool player Trent Alexander-Arnold.
Sentence 3: Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.
  Fact 3.0: Trent John Alexander-Arnold was born on 7 October 1998.
  Fact 3.1: Trent John Alexander-Arnold is English.
  Fact 3.2: Trent John Alexander-Arnold is a professional footballer.
  Fact 3.3: Trent John Alexander-Arnold plays as a right-back.
  Fact 3.4: Trent John Alexander-Arnold plays for Premier League club Liverpool.

Retrieved items:
<item 1>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 1>

<item 2>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 2>

<item 3>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 3>

<item 4>
Nick Powell
Nicholas Edward "Nick" Powell (born 23 March 1994) is an English professional footballer who plays for Wigan Athletic as a midfielder.

 He began his career at Crewe Alexandra, initially as a forward, making his debut at the age of 16.

 After impressing for Crewe Alexandra during the 2011–12 season, including scoring in their 2–0 victory in the League Two play-off Final, he moved to Manchester
United in July 2012.

 Unable to break into the Manchester United first team, however, he went on loan to Wigan Athletic, Leicester City and Hull City before being released in June
2016.

 He joined Wigan Athletic in July 2016.

 He has represented England at under-16, under-17, under-18, under-19 and under-21 levels.
</item 4>

<item 5>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 5>

<item 6>
Trent Alexander-Arnold
Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.

 He is an academy graduate of Liverpool and made his senior debut for the club in 2016, aged 18.

 The following year, he won the club's Young Player of the Season award.

 He has also represented England at all youth levels from under-16 through to under-19.

 Due to the length of his name, his surname is shortened to Alexander for Premier League fixtures.
</item 6>

<item 7>
John Logan (footballer, born 1912)
John William Logan (16 August 1912 – October 1980) was an English professional footballer born in Horden, near Peterlee, County Durham who played in the
Football League for Darlington, Barnsley and Sheffield Wednesday.

 He played as a wing half.
</item 7>

<item 8>
Tommy Bell (footballer, born 1906)
Thomas Bell (9 November 1906 – 1983) was an English professional footballer born in Seaham Harbour.

 He could play at inside right, centre forward and right half and was a regular goalscorer throughout his professional career.
</item 8>

<item 9>
John Alexander (footballer, born 1955)
John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.

 Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in
1976.

 After scoring twice in 15 appearances over the course of two years with Millwall, he moved to Reading; he enjoyed greater success with Reading, scoring nine
goals in 25 games.

 In 1981, he joined Northampton Town, but spent just one season with them before retiring from football at the age of 26 in 1982.

 Upon retiring from football, Alexander got a job with the BBC, but he later returned to football as club secretary at Watford.

 In 2000, he took up the same post at Tottenham Hotspur, spending 10 years there before applying for the same job at Manchester United, where he would replace
the retiring Ken Ramsden.

 He took over at Manchester United on 1 July 2010.

 He is the uncle of Liverpool player Trent Alexander-Arnold.
</item 9>

<item 10>
Wilton Lines
Wilton Lines (fl.

 1889–1890) was an English professional footballer born in Birmingham.

 Lines played seven games for Small Heath in the inaugural 1889–90 season of the Football Alliance, covering a variety of forward positions, but was unable to
displace the established forwards.
</item 10>
=== END ITEM id=695d717be16be632 ===

=== ITEM id=1aa763395c4b841f ===
Gold sentences and facts:
Sentence 0:  It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
  Fact 0.0: The work is the third installment of the 'Naked' series.
  Fact 0.1: 'Naked Killer' is a work in the 'Naked' series released in 1992.
  Fact 0.2: 'Naked Weapon' is a work in the 'Naked' series released in 2002.
Sentence 1: Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
  Fact 1.0: Naked Weapon is also titled 赤裸特工.
  Fact 1.1: Naked Weapon is a 2002 film.
  Fact 1.2: Naked Weapon is a Hong Kong film.
  Fact 1.3: Naked Weapon is an action-thriller film.
  Fact 1.4: Naked Weapon was directed by action choreographer Tony Ching.
  Fact 1.5: Naked Weapon stars Maggie Q.
  Fact 1.6: Naked Weapon stars Anya Wu.
  Fact 1.7: Naked Weapon stars Daniel Wu.

Retrieved items:
<item 1>
The Naked Gun 2½: The Smell of Fear
The Naked Gun 2½: The Smell of Fear is a 1991 comedy film.

 It is the sequel to the 1988 film "" and the second installment in "The Naked Gun" film series.

 The film stars Leslie Nielsen as the comically bumbling Police Lt. Frank Drebin of "Police Squad!

".

 Priscilla Presley plays the role of Jane, with O. J. Simpson as Nordberg and George Kennedy as police captain Ed Hocken.

 The film also features Robert Goulet (who previously made a "special guest star" appearance on "Police Squad!")

 as the villainous Quentin Hapsburg and Richard Griffiths as renewable fuel advocate Dr. Albert S. Meinheimer (as well as his evil double, Earl Hacker).

 Zsa Zsa Gabor, Mel Tormé and members of the Chicago Bears have cameo roles.
</item 1>

<item 2>
Naked Weapon
Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
</item 2>

<item 3>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 3>

<item 4>
Mel Gibson filmography
Five years later he played Martin Riggs in the buddy cop action comedy "Lethal Weapon" alongside Danny Glovera role he later reprised in its sequels "Lethal
Weapon 2" (1989), "Lethal Weapon 3" (1992), and "Lethal Weapon 4" (1998).
</item 4>

<item 5>
Naked Soldier
Naked Soldier (絕色武器) is a 2012 Hong Kong action film directed by Marco Mak and starring Jennifer Tse.

 It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
</item 5>

<item 6>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 6>

<item 7>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 7>

<item 8>
Naked Gun 33⅓: The Final Insult
Naked Gun 33⅓: The Final Insult is a 1994 comedy film, and the third and final installment in "The Naked Gun" film series, which was based on the "Police Squad!

" television series.
</item 8>

<item 9>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 9>

<item 10>
American Pie Presents: The Naked Mile
American Pie Presents: The Naked Mile (also known as American Pie: The Naked Mile) is a 2006 American sex comedy film released by Universal Pictures.

 It is the second installment in the "American Pie Presents" series and the fifth installment in the "American Pie" franchise.

 The film begins a story arc that concludes with "" (2007).

 John White stars as Erik Stifler, a high school senior who is given a "guilt free pass" by his girlfriend, Tracy Sterling (Jessy Schram), and so visits the
Beta House fraternity led by his cousin, Dwight Stifler (Steve Talley), to run a mile naked.

 Christopher McDonald co-stars as Erik's father, Harry, and Eugene Levy once again plays Jim's Dad, who turns out to be a family friend of both Erik's and
Tracy's.

 Also, it is in this film that his name is revealed to be "Noah Levenstein".
</item 10>
=== END ITEM id=1aa763395c4b841f ===

=== ITEM id=13a173157fee2f9c ===
Gold sentences and facts:
Sentence 0: Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York, United States.
  Fact 0.0: Albany International Airport has the IATA code ALB.
  Fact 0.1: Albany International Airport has the ICAO code KALB.
  Fact 0.2: Albany International Airport has the FAA LID ALB.
  Fact 0.3: Albany International Airport is a public airport.
  Fact 0.4: Albany International Airport is seven miles (11 km) northwest of Albany.
  Fact 0.5: Albany International Airport is in Albany County, New York, United States.
Sentence 1: Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown Madison, the capital of Wisconsin.
  Fact 1.0: Dane County Regional Airport (DCRA) has the IATA code MSN.
  Fact 1.1: Dane County Regional Airport (DCRA) has the ICAO code KMSN.
  Fact 1.2: Dane County Regional Airport (DCRA) has the FAA LID MSN.
  Fact 1.3: Dane County Regional Airport is also known as Truax Field.
  Fact 1.4: Dane County Regional Airport is a civil-military airport.
  Fact 1.5: Dane County Regional Airport is located six miles northeast of downtown Madison.
  Fact 1.6: Madison is the capital of Wisconsin.

Retrieved items:
<item 1>
Air Wisconsin
Air Wisconsin Airlines is a regional airline based at Appleton International Airport in the town of Greenville, Wisconsin, United States, near Appleton.

 Air Wisconsin is the largest privately held regional airline in the United States.

 It currently operates regional jet flights as American Eagle under contract to American Airlines via a code sharing agreement, serving cities in the U.S. and
Canada with hubs at Boston Logan International Airport (BOS), Charlotte Douglas International Airport (CLT), Philadelphia International Airport (PHL) and Ronald
Reagan Washington National Airport (DCA) .

 Air Wisconsin previously operated United Express service on behalf of United Airlines followed by US Airways Express service on behalf of US Airways prior to
becoming an American Eagle regional air carrier.

 The company has announced it will once again partner with United Airlines as a United Express code sharing air carrier with primary hubs to be located at
Chicago O'Hare International Airport (ORD) and Washington Dulles International Airport (IAD) beginning in September 2017 and will be "exclusively operating" as
United Express by March 2018.
</item 1>

<item 2>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 2>

<item 3>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 3>

<item 4>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 4>

<item 5>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 5>

<item 6>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 6>

<item 7>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 7>

<item 8>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 8>

<item 9>
Truax Field Air National Guard Base
Truax Field Air National Guard Base (IATA: MSN, ICAO: KMSN, FAA LID: MSN) , also known as Truax Field, is a military facility located at Dane County Regional
Airport.

 It is located five miles (8 km) northeast of the center of Madison, a city in Dane County, Wisconsin, United States.
</item 9>

<item 10>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 10>
=== END ITEM id=13a173157fee2f9c ===

=== ITEM id=e13469b9eab8fdb8 ===
Gold sentences and facts:
Sentence 0: McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.
  Fact 0.0: McFarland, USA is also known as McFarland.
  Fact 0.1: McFarland, USA is a 2015 American sports drama film.
  Fact 0.2: McFarland, USA was directed by Niki Caro.
  Fact 0.3: McFarland, USA was produced by Mark Ciardi.
  Fact 0.4: McFarland, USA was produced by Gordon Gray.
  Fact 0.5: McFarland, USA was written by Christopher Cleveland.
  Fact 0.6: McFarland, USA was written by Bettina Gilois.
  Fact 0.7: McFarland, USA was written by Grant Thompson.
  Fact 0.8: McFarland, USA has music composed by Antônio Pinto.
Sentence 1: The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.
  Fact 1.0: The Black Hole is a 1979 American science fiction film.
  Fact 1.1: The Black Hole was directed by Gary Nelson.
  Fact 1.2: The Black Hole was produced by Walt Disney Productions.

Retrieved items:
<item 1>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 1>

<item 2>
Full Metal Jacket
Full Metal Jacket is a 1987 British-American war film directed and produced by Stanley Kubrick.

 The screenplay by Kubrick, Michael Herr, and Gustav Hasford was based on Hasford's novel "The Short-Timers" (1979).

 Its storyline follows a platoon of U.S. Marines through their training, primarily focusing on two privates, Joker and Pyle, who struggle to get through camp
under their foul-mouthed drill instructor, Gunnery Sergeant Hartman, and the experiences of two of the platoon's Marines in the Tet Offensive during the Vietnam
War.

 The film's title refers to the full metal jacket bullet used by soldiers.

 The film was released in the United States on June 26, 1987.
</item 2>

<item 3>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 3>

<item 4>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 4>

<item 5>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 5>

<item 6>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 6>

<item 7>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 7>

<item 8>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 8>

<item 9>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 9>

<item 10>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 10>
=== END ITEM id=e13469b9eab8fdb8 ===

=== ITEM id=6343905662831fbc ===
Gold sentences and facts:
Sentence 0: Operalia, The World Opera Competition is an annual international competition for young opera singers.
  Fact 0.0: Operalia, The World Opera Competition is an annual international competition.
  Fact 0.1: Operalia is a competition for young opera singers.
Sentence 1:  Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti, Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
  Fact 1.0: Operalia was founded in 1993.
  Fact 1.1: Operalia was founded by Plácido Domingo.
  Fact 1.2: Operalia has helped launch the careers of several important artists.
  Fact 1.3: Operalia has helped launch the career of Joseph Calleja.
  Fact 1.4: Operalia has helped launch the career of Giuseppe Filianoti.
  Fact 1.5: Operalia has helped launch the career of Rolando Villazón.
  Fact 1.6: Operalia has helped launch the career of José Cura.
  Fact 1.7: Operalia has helped launch the career of Joyce DiDonato.
  Fact 1.8: Operalia has helped launch the career of Elizabeth Futral.
  Fact 1.9: Operalia has helped launch the career of Inva Mula.
  Fact 1.10: Operalia has helped launch the career of Ana María Martínez.
  Fact 1.11: Operalia has helped launch the career of Sonya Yoncheva.
Sentence 2: Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
  Fact 2.0: Sonya Yoncheva was born on 25 December 1981.
  Fact 2.1: Sonya Yoncheva’s Bulgarian name is Соня Йончева.
  Fact 2.2: Sonya Yoncheva is a Bulgarian operatic soprano.

Retrieved items:
<item 1>
Guanqun Yu
Guanqun Yu (; born 1982) is a Chinese soprano who has sung in opera houses and concert halls internationally.

 In 2008 she won the Belvedere International Singing Competition and in 2012 she placed 2nd in the Operalia, The World Opera Competition.

 She is particularly known for portraying heroines in the operas of Wolfgang Amadeus Mozart, Giacomo Puccini, and Giuseppe Verdi.
</item 1>

<item 2>
Maria Katzarava
Maria Katzarava (born 1984) is an opera singer from Mexico who achieved international fame by winning first place in the Operalia competition in the opera and
zarzuela categories.

 Prior to this, she came to the attention of notable Mexican tenor Ramón Vargas, who helped her get a scholarship, and then won Mexico's Carlo Morelli national
singing competition in 2005, which earned her a debut at the Palacio de Bellas Artes.

 Since winning the Operalia competition, Katzarava has performed in Europe, where she currently lives, Asia and the Americas.

 She currently trains with soprano Mirella Freni.
</item 2>

<item 3>
Andrea Carroll (soprano)
Andrea Carroll is an American soprano who has had an active international career in concerts and operas since 2012.

 A finalist in the 2015 Operalia, The World Opera Competition, she is particularly known for her performances with the Houston Grand Opera and the Vienna State
Opera.
</item 3>

<item 4>
Ao Li
Ao Li (born 8 February 1988) is a Chinese operatic bass-baritone and voice teacher who is particularly known for his performances at the San Francisco Opera.

 In 2013 he won first prize in the Operalia, The World Opera Competition and in 2014 he won the Metropolitan Opera National Council Auditions.
</item 4>

<item 5>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 5>

<item 6>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 6>

<item 7>
Rachel Willis-Sørensen
Rachel Willis-Sørensen (born 1984) is an American operatic soprano.

 She was a winner of the Metropolitan Opera National Council Auditions in 2010.

 In 2014, she won first prize, the Birgit Nilsson Prize, and the Zarzuela Prize of the Operalia competition.

 Other awards include first place in the 2009 Eleanor McCollum Competition for Young Singers in Houston and the 2010 Sara Tucker Grant from the Richard Tucker
Music Foundation.
</item 7>

<item 8>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 8>

<item 9>
Norine Burgess
Norine Burgess is a Canadian singer.

 She is a graduate of the University of Calgary and the University of Toronto’s Opera School, mezzo-soprano.

 She received additional training as a member of the Canadian Opera Company (COC) Ensemble where she appeared in Electra, Suor Angelica, Lulu and Der
Rosenkavalier.

 Ms. Burgess also performed Le Nozze di Figaro(Cherubino), Ariadne auf Naxos (Dryad), La Traviata(Flora) and Die Zauberflöte (Second Lady) with the COC.

 Additional operatic successes include Le Nozze di Figaro (Cherubino) and Carmen (Mercedes) with the Vancouver Opera, Fenena in Nabucco with Manitoba Opera,
Albert Herring (Nancy) with the Calgary Opera and many appearances with the Edmonton Opera.
</item 9>

<item 10>
Jane Rhodes
Jane Marie Andrée Rhodes (March 13, 1929 – May 7, 2011) was a French opera singer whose voice encompassed both the soprano and high mezzo-soprano ranges.

 Her most celebrated role was Carmen, which she sang in the opera's first ever staging at the Palais Garnier.

 She also created the roles of Isadora in Marcel Landowski's "Le Fou" and Maguelone in Georges Delerue's "Le Chevalier de Neige" and sang Renata in the first
recorded performance of Prokofiev's "The Fiery Angel".

 Admired for both her voice and her glamorous stage presence, she was nicknamed the "Bardot of the Opéra".

 She was married to conductor Roberto Benzi.
</item 10>
=== END ITEM id=6343905662831fbc ===

=== ITEM id=e396ddb59478d1b0 ===
Gold sentences and facts:
Sentence 0: Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.
  Fact 0.0: Samantha Cristoforetti was born 26 April 1977 in Milan.
  Fact 0.1: Samantha Cristoforetti is an Italian European Space Agency astronaut.
  Fact 0.2: Samantha Cristoforetti is an Italian Air Force pilot.
  Fact 0.3: Samantha Cristoforetti is an Italian engineer.
Sentence 1: Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force), and a former CNES spationaut.
  Fact 1.0: Jean-Loup Jacques Marie Chrétien was born 20 August 1938.
  Fact 1.1: Jean-Loup Jacques Marie Chrétien is French.
  Fact 1.2: Jean-Loup Jacques Marie Chrétien is retired.
  Fact 1.3: Jean-Loup Jacques Marie Chrétien holds the rank of Général de Brigade (brigadier general).
  Fact 1.4: Jean-Loup Jacques Marie Chrétien served as a general in the Armée de l'Air (French air force).
  Fact 1.5: Jean-Loup Jacques Marie Chrétien is a former CNES spationaut.

Retrieved items:
<item 1>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 1>

<item 2>
Jean-Loup Puget
Jean-Loup Puget (born 7 March 1947) is a French astrophysicist.

 His current research interests lie in the Cosmic Microwave Background.

 Jean-Loup Puget and his collaborators reported the first identification of the Cosmic infrared background using COBE data.

 He is also, along with Alain Léger, credited with the origin of the hypothesis that the series of infrared lines observed in numerous astrophysical objects are
caused by emission from polycyclic aromatic hydrocarbons.

 He is currently principal investigator of the HFI module of the Planck space mission.
</item 2>

<item 3>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 3>

<item 4>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 4>

<item 5>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 5>

<item 6>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 6>

<item 7>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 7>

<item 8>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 8>

<item 9>
Soyuz T-6
Soyuz T-6 was a manned spaceflight to Earth orbit to the Salyut 7 space station in 1982.

 Along with two Soviet cosmonauts, the crew included a Frenchman, Jean-Loup Chrétien.
</item 9>

<item 10>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 10>
=== END ITEM id=e396ddb59478d1b0 ===

=== ITEM id=52e9048ff2d18c8b ===
Gold sentences and facts:
Sentence 0: Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
  Fact 0.0: Alan Crosland was born on August 10, 1894.
  Fact 0.1: Alan Crosland died on July 16, 1936.
  Fact 0.2: Alan Crosland was an American stage actor.
  Fact 0.3: Alan Crosland was an American film director.
Sentence 1: Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.
  Fact 1.0: Kurt Voss was born Kurt Christopher Peter Wössner.
  Fact 1.1: Kurt Voss is an American film director.
  Fact 1.2: Kurt Voss is an American screenwriter.
  Fact 1.3: Kurt Voss is an American musician-songwriter.

Retrieved items:
<item 1>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 1>

<item 2>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 2>

<item 3>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 3>

<item 4>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 4>

<item 5>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 5>

<item 6>
Al Ulmer
Alfred C. Ulmer, Jr. was born in Jacksonville, Florida in August 1916.

 He was of Swiss extraction on his father's side, his father having been born in Zurich.

 Ulmer graduated from Princeton University in 1939 and joined the United States Navy prior to the start of World War II, ultimately becoming a major head of
intelligence operations during World War II.

 He married Doris Gibson Bridges and had three sons (Alfred III, James and Nicolas) and a daughter (Marguerite).

 He received the Intelligence Medal of Merit when he retired from his position in 1962.

 Ulmer then went on to business and in the 1980s joined the Swiss banking firm Lombard Odier et Cie. in Geneva, later setting up Lombard Odier's operations in
Bermuda.

 He died on June 22, 2000 in Virginia Beach, Virginia.
</item 6>

<item 7>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 7>

<item 8>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 8>

<item 9>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 9>

<item 10>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 10>
=== END ITEM id=52e9048ff2d18c8b ===

=== ITEM id=0ef3f23bd653631d ===
Gold sentences and facts:
Sentence 0: John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.
  Fact 0.0: John Kingsley Orton, known as Joe Orton, was born on 1 January 1933.
  Fact 0.1: Joe Orton died on 9 August 1967.
  Fact 0.2: Joe Orton was an English playwright.
  Fact 0.3: Joe Orton was an English author.
Sentence 1:  His public career was short but prolific, lasting from 1964 until his death three years later.
  Fact 1.0: Joe Orton's public career was short but prolific.
  Fact 1.1: Joe Orton's public career lasted from 1964 until his death.
  Fact 1.2: Joe Orton died three years after his public career began in 1964.
Sentence 2:  During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.
  Fact 2.0: During his brief public career, Joe Orton shocked audiences with his scandalous black comedies.
  Fact 2.1: During his brief public career, Joe Orton outraged audiences with his scandalous black comedies.
  Fact 2.2: During his brief public career, Joe Orton amused audiences with his scandalous black comedies.
Sentence 3:  The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
  Fact 3.0: The adjective 'Ortonesque' is sometimes used to refer to work characterised by a dark yet farcical cynicism.
  Fact 3.1: The cynicism is similarly dark yet farcical.
Sentence 4: Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary modernism, and is considered by many to be among the greatest novelists in the English language.
  Fact 4.0: Henry James, OM, was an American author.
  Fact 4.1: Henry James was born in 1843.
  Fact 4.2: Henry James died in 1916.
  Fact 4.3: Henry James was regarded as a key transitional figure between literary realism and literary modernism.
  Fact 4.4: Henry James is considered by many to be among the greatest novelists in the English language.
Sentence 5:  He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
  Fact 5.0: Henry James was the son of Henry James, Sr.
  Fact 5.1: Henry James was the brother of the renowned philosopher and psychologist William James.
  Fact 5.2: Henry James was the brother of diarist Alice James.

Retrieved items:
<item 1>
Prick Up Your Ears
Prick Up Your Ears is a 1987 British film, directed by Stephen Frears, about the playwright Joe Orton and his lover Kenneth Halliwell.

 The screenplay was written by Alan Bennett, based on the biography by John Lahr.

 The film stars Gary Oldman as Orton, Alfred Molina as Halliwell, Wallace Shawn as Lahr and Vanessa Redgrave as Peggy Ramsay.
</item 1>

<item 2>
What the Butler Saw (play)
What the Butler Saw is a farce written by the English playwright Joe Orton.

 It was premièred at the Queen's Theatre in London on 5 March 1969.

 It was Orton's final play and the second to be performed after his death, following "Funeral Games" in 1968.
</item 2>

<item 3>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 3>

<item 4>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 4>

<item 5>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 5>

<item 6>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 6>

<item 7>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 7>

<item 8>
The Ruffian on the Stair
The Ruffian On the Stair is a play by British playwright Joe Orton which was first broadcast on BBC Radio in August 1964.

 It is an unsympathetic yet comedic one-act portrayal of working class England, as played out by a couple and a mysterious young man who toys with their lives.

 It was based on "The Boy Hairdresser", a novel by Orton and his lover Kenneth Halliwell.

 The title and play are based on a few lines from poet and dramatist William Ernest Henley: "Madam Life's a piece in bloom, / Death goes dogging everywhere: /
She's the tenant of the room, / He's the ruffian on the stair."

 "Ruffian" is not as renowned as other works such as "Loot" and "What the Butler Saw", but it is still staged on occasion.
</item 8>

<item 9>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 9>

<item 10>
Prick Up Your Ears (play)
Prick Up Your Ears is a play by Simon Bent, based on the life of playwright Joe Orton.

 Produced by Sonia Friedman it opened at the Comedy Theatre in London's West End on 30 September 2009 following previews from 17 September.

 It starred Chris New as Joe Orton and Matt Lucas as Orton's lover and murderer, Kenneth Halliwell.
</item 10>
=== END ITEM id=0ef3f23bd653631d ===

=== ITEM id=504aa909c36d9811 ===
Gold sentences and facts:
Sentence 0: Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.
  Fact 0.0: Disteghil Sar is also called Distaghil Sar.
  Fact 0.1: Disteghil Sar is the highest mountain in the Hispar Muztagh.
  Fact 0.2: The Hispar Muztagh is a subrange of the Karakoram mountain range.
  Fact 0.3: Disteghil Sar is in Gilgit-Baltistan.
Sentence 1:  The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
  Fact 1.0: Disteghil Sar has a top ridge about 3 km long.
  Fact 1.1: The top ridge of Disteghil Sar is above 7400 m.
  Fact 1.2: Disteghil Sar has three distinct summits.
  Fact 1.3: The (north)west summit of Disteghil Sar is 7885 m high.
  Fact 1.4: The central summit of Disteghil Sar is 7760 m high.
  Fact 1.5: The (south)east summit of Disteghil Sar is 7696 m or 7535 m high.
Sentence 2:  Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
  Fact 2.0: The summit of Disteghil Sar has an elevation of 7295 m.
  Fact 2.1: Disteghil Sar is the highest peak in the Yengisogat subrange of the Karakoram.

Retrieved items:
<item 1>
Yengisogat
The Yengisogat range (音苏盖提 ), also known as the Wesm Mountains, is a Chinese subrange of the Karakoram mountain range.

 It lies north of the Baltoro Muztagh, home of the eight-thousanders of the Karakoram.

 The highest peak is Huangguan Shan, or Crown Peak, 7,265 m (23,835 ft) (also sometimes given as 7,295 m/23,934 ft).
</item 1>

<item 2>
Kezhen Peak
Kezhen Peak, also known as Karpogo Sar, is a mountain in the Karakoram mountain range.

 It is located in the Xinjiang Autonomous Region of China.
</item 2>

<item 3>
Kanjut Sar
Kanjut Sar (Urdu: کنجت سر‎ ) or Kunjudh Sar as pronounced in "Wakhi" is a mountain located in the Hispar Muztagh, a subrange of the Karakoram mountain range.

 Kunjudh Sar in wakhi language mean that which overlooks Kunjudh, or above Kunjudh, while Khujudh is the wakhi name for Lower Hunza.

 It is the 26th highest mountain on Earth and the 11th highest in Pakistan.
</item 3>

<item 4>
The Crown (mountain)
The Crown, also known as Huang Guan Shan and sometimes Crown Peak, is a mountain in the Karakoram mountain range in China.

 It is located in the Xinjiang Uyghur Autonomous Region of China.

 Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
</item 4>

<item 5>
Momhil Sar
Momhil Sar, or Mumhail Sar as pronounced in Wakhi is at 7343 m above sea level, is the 64th highest mountain peak in the world.

 Mumhail Sar in Wakhi means the mountain that overlooks or is above Grandmother's cattle pen or paddock.

 It is situated in the Hispar Muztagh subrange of the Karakoram range, a few kilometres to the north-west of its parent peak Trivor.
</item 5>

<item 6>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 6>

<item 7>
Passu Sar
Passu Sar (Urdu: ‎ ; or Passu Sar, Passu I) is a mountain peak in the Batura Muztagh, a sub-range of the Karakoram mountain range, located in the Gilgit
District of Gilgit-Baltistan, Pakistan, west of the Hunza Valley.

 It is the high point of the Passu massif, which also includes Passu Diar (or "Passu East", "Pasu II").

 The peak lies on the main ridge of the Batura Muztagh, about 7 km (4 mi) east of Batura Sar.
</item 7>

<item 8>
Yutmaru Sar
Yutmaru Sar is a mountain in the Hispar mountain range, a subrange of the Karakoram.

 At an elevation of 7283 m it is the 88th highest mountain in the world.

 Yutmaru Sar is located in Gilgit-Baltistan, Pakistan.

 It was first climbed in 1980.
</item 8>

<item 9>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 9>

<item 10>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 10>
=== END ITEM id=504aa909c36d9811 ===

=== ITEM id=21fc12c5eb63ebde ===
Gold sentences and facts:
Sentence 0: Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
  Fact 0.0: Wilmington International Airport has the IATA code ILM.
  Fact 0.1: Wilmington International Airport has the ICAO code KILM.
  Fact 0.2: Wilmington International Airport has the FAA LID ILM.
  Fact 0.3: Wilmington International Airport is a public airport.
  Fact 0.4: Wilmington International Airport is located just north of Wilmington, North Carolina.
  Fact 0.5: Wilmington International Airport is located in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
Sentence 1: Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km) northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.
  Fact 1.0: Texarkana Regional Airport has the IATA code TXK.
  Fact 1.1: Texarkana Regional Airport has the ICAO code KTXK.
  Fact 1.2: Texarkana Regional Airport has the FAA LID TXK.
  Fact 1.3: Texarkana Regional Airport is also known as Webb Field.
  Fact 1.4: Texarkana Regional Airport is a public use airport.
  Fact 1.5: Texarkana Regional Airport is located three nautical miles (6 km) northeast of the central business district of Texarkana.
  Fact 1.6: Texarkana is a city in Miller County, Arkansas, United States.

Retrieved items:
<item 1>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 1>

<item 2>
Pensacola International Airport
Pensacola International Airport (IATA: PNS, ICAO: KPNS, FAA LID: PNS) , formerly Pensacola Gulf Coast Regional Airport and Pensacola Regional Airport (Hagler
Field), is a public use airport three nautical miles (6 km) northeast of the central business district of Pensacola, in Escambia County, Florida, United States.

 It is owned by the City of Pensacola.

 Despite the name, this airport does not offer direct international flights.

 This airport is one of the five major airports in North Florida, others being: Northwest Florida Beaches International Airport, Destin-Fort Walton Beach
Airport Tallahassee International Airport, and Jacksonville International Airport.
</item 2>

<item 3>
Elizabeth City Regional Airport
Elizabeth City Regional Airport (IATA: ECG, ICAO: KECG, FAA LID: ECG) is a joint civil-military public and military use airport located three nautical miles
(6 km) southeast of the central business district of Elizabeth City, in Pasquotank County, North Carolina, United States.

 The airport, on the shore of the Pasquotank River, is also known as Elizabeth City-Pasquotank County Regional Airport or ECG Regional Airport.

 It is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "general aviation" facility.
</item 3>

<item 4>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 4>

<item 5>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 5>

<item 6>
Wilmington International Airport
Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated
Wrightsboro, Cape Fear Township, New Hanover County.
</item 6>

<item 7>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 7>

<item 8>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 8>

<item 9>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 9>

<item 10>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 10>
=== END ITEM id=21fc12c5eb63ebde ===
