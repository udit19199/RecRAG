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

This file has 11 items.

=== ITEM id=67340e51d148fc20 ===
Gold sentences and facts:
Sentence 0: The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).
  Fact 0.0: The Washington City Canal operated from 1815 until the mid-1850s.
  Fact 0.1: The Washington City Canal operated in Washington, D.C.
  Fact 0.2: The Washington City Canal connected the Anacostia River to Tiber Creek.
  Fact 0.3: The Washington City Canal connected the Anacostia River to the Potomac River.
  Fact 0.4: The Washington City Canal later connected the Anacostia River to the Chesapeake and Ohio Canal (C&O).
  Fact 0.5: The Anacostia River was called the 'Eastern Branch' at that time.
Sentence 1: The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River to Hagerstown, Indiana.
  Fact 1.0: The Whitewater Canal was built between 1836 and 1847.
  Fact 1.1: The Whitewater Canal spanned a distance of seventy-six miles.
  Fact 1.2: The Whitewater Canal stretched from Lawrenceburg, Indiana.
  Fact 1.3: Lawrenceburg, Indiana is on the Ohio River.
  Fact 1.4: The Whitewater Canal stretched to Hagerstown, Indiana.

Retrieved items:
<item 1>
Cincinnati and Whitewater Canal Tunnel
The Cincinnati and Whitewater Canal Tunnel is a historic yet abandoned canal tunnel in the southwestern corner of the U.S. state of Ohio.

 Located within the village of Cleves near Cincinnati, it was constructed in 1837 as part of the Whitewater Canal system.

 Since the canal's closure, it has largely been forgotten, but it has been designated a historic site.
</item 1>

<item 2>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 2>

<item 3>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 3>

<item 4>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 4>

<item 5>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 5>

<item 6>
Whitewater Canal
The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River
to Hagerstown, Indiana.
</item 6>

<item 7>
Tiber Creek
Tiber Creek or Tyber Creek is a tributary of the Potomac River in Washington, D.C. In the 19th century it was modified to become part of the Washington City
Canal, and subsequently was enclosed in a tunnel.
</item 7>

<item 8>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 8>

<item 9>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 9>

<item 10>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 10>
=== END ITEM id=67340e51d148fc20 ===

=== ITEM id=33fadeaf622a10d7 ===
Gold sentences and facts:
Sentence 0: Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
  Fact 0.0: Kelechukwu 'Kele' Okereke was born on 13 October 1981.
  Fact 0.1: Kelechukwu 'Kele' Okereke is also known by the mononym Kele.
  Fact 0.2: Kelechukwu 'Kele' Okereke is a British musician.
  Fact 0.3: Kelechukwu 'Kele' Okereke is best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
Sentence 1: Jon Hume (born 6 November 1983) is the lead singer of Evermore.
  Fact 1.0: Jon Hume was born on 6 November 1983.
  Fact 1.1: Jon Hume is the lead singer of Evermore.
Sentence 2:  Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
  Fact 2.0: Jon is the eldest of the three brothers.
  Fact 2.1: The three brothers make up the New Zealand band Evermore.
  Fact 2.2: Evermore is a New Zealand band.

Retrieved items:
<item 1>
Between the Lines (Evermore song)
"Between the Lines" is the first single by Evermore, taken from their third studio album "".

 Evermore's Jon Hume said that "Between the Lines was the first song to come out of a search for a new musical experience as a band."

 It was released as a free download on Evermore's official website on 10 November 2008.
</item 1>

<item 2>
Dann Hume
Dann Hume (born Daniel Benjamin Cobbe, 1 September 1987, Whangaparaoa, New Zealand) is a singer-songwriter and record producer.

 Hume is the youngest of three brothers (with Peter Hume and Jon Hume) who make up the alternative rock band, Evermore since 1999.
</item 2>

<item 3>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 3>

<item 4>
Evermore (Evermore album)
Evermore is the self-titled international debut album by New Zealand rock band Evermore, which was also released in Australia and New Zealand on 12 March 2010.

 The album is a collection of songs from the band's entire history, as well as three new songs including the single "Underground".
</item 4>

<item 5>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 5>

<item 6>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 6>

<item 7>
Running (Evermore song)
"Running" is the lead single by alternative rock band Evermore, taken from their second studio album, "Real Life" (July 2006).

 It was written by backing vocalist and drummer, Dann Hume.

 It was issued on 3 June, ahead of the album.

 It was co-produced by Jon with John Alagía (Dave Matthews Band, John Mayer).

 It peaked at No. 4 on their native RIANZ Singles Chart in New Zealand and No. 5 on their adopted country of Australia's ARIA Singles Chart.
</item 7>

<item 8>
Kele Okereke
Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of
the indie rock band Bloc Party.
</item 8>

<item 9>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 9>

<item 10>
Jon Hume
Jon Hume (born 6 November 1983) is the lead singer of Evermore.

 He is a singer-songwriter, multi-instrumentalist, Record Producer and Studio owner.

 Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
</item 10>
=== END ITEM id=33fadeaf622a10d7 ===

=== ITEM id=bedd050edd8f1cf7 ===
Gold sentences and facts:
Sentence 0: Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.
  Fact 0.0: Edward Sharpe and the Magnetic Zeros is an American folk rock band.
  Fact 0.1: Edward Sharpe and the Magnetic Zeros was formed in Los Angeles, California.
  Fact 0.2: Edward Sharpe and the Magnetic Zeros was formed in 2007.
Sentence 1: Louis XIV is an American rock band from San Diego, California.
  Fact 1.0: Louis XIV is an American rock band.
  Fact 1.1: Louis XIV is from San Diego, California.
Sentence 2:  The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.
  Fact 2.0: The band released four EPs between 2003 and 2007.
  Fact 2.1: The band released three albums between 2003 and 2008.
  Fact 2.2: Two of the band's three albums, the latter two, were distributed by Atlantic Records.

Retrieved items:
<item 1>
Edward Sharpe and the Magnetic Zeros
Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.

 The group is led by lead singer Alex Ebert.

 The band's name is based on a story Ebert wrote in his youth, about a messianic figure named Edward Sharpe.

 Drawing from roots rock, folk, gospel, and psychedelic music, the band's image and sound evoke the hippie movement of the 1960s and 1970s.

 The group's first show was played July 18, 2007 at The Troubadour in West Hollywood, California.

 Their first studio album, "Up from Below", was released on July 7, 2009 on Community Records and featured the popular single "Home".

 The group released their second full-length album, "Here", on May 29, 2012, and third album, "Edward Sharpe and the Magnetic Zeros", on July 23, 2013.

 Their fourth studio album, "PersonA", was released in April 2016.
</item 1>

<item 2>
Alexander (Alex Ebert album)
Alexander is the first solo album by American musician Alex Ebert, lead singer of Edward Sharpe and the Magnetic Zeros and Ima Robot.

 It was released on March 1, 2011 under the Community Records label.

 The song "Truth" was featured in the season 4 premiere of AMC's "Breaking Bad" titled "Box Cutter".
</item 2>

<item 3>
Edward Sharpe and the Magnetic Zeros (album)
Edward Sharpe and the Magnetic Zeros is the third album by Edward Sharpe and the Magnetic Zeros.

 It was released on July 23, 2013 in North America and was released on July 29, 2013 around the world through Vagrant Records, Rough Trade Records and Communion
Records.

 Frontman Alex Ebert stated that "These songs mean everything to me - It's the rawest, most liberated, most rambunctious stuff we've done."
</item 3>

<item 4>
PersonA
PersonA (pronounced "Persona") is the fourth studio album by the band Edward Sharpe and the Magnetic Zeros.

 It was released on April 15, 2016.
</item 4>

<item 5>
Orpheo McCord
Orpheo McCord (born 1979) is an American drummer and percussionist.

 He is an original member of Edward Sharpe and the Magnetic Zeros, and a former member of The Fall, Fool's Gold and Cass McCombs.
</item 5>

<item 6>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 6>

<item 7>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 7>

<item 8>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 8>

<item 9>
Louis XIV (band)
Louis XIV is an American rock band from San Diego, California.

 The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.

 The band broke up in 2009, but in 2013, in an interview with The Reno Dispatch, Jason Hill confirmed that the band had decided to reunite.
</item 9>

<item 10>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 10>
=== END ITEM id=bedd050edd8f1cf7 ===

=== ITEM id=b71f9616ab2f99ac ===
Gold sentences and facts:
Sentence 0: Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway 329 (also known as the Avery Island Highway).
  Fact 0.0: Rynella is an unincorporated community.
  Fact 0.1: Rynella is in Iberia Parish, Louisiana.
  Fact 0.2: Rynella consists of a few residential dwellings.
  Fact 0.3: Rynella has a volunteer fire department.
  Fact 0.4: The residential dwellings and the volunteer fire department are along Highway 329.
  Fact 0.5: Highway 329 is also known as the Avery Island Highway.
Sentence 1:  The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA" McIlhenny (someone eventually adding an extra L to the resulting word).
  Fact 1.0: The name Rynella allegedly is an acronym.
  Fact 1.1: The acronym derives from the last two letters of the first names of three local women.
  Fact 1.2: The three local women are Rosema, Pauli, and Lei McIlhenny.
  Fact 1.3: The acronym uses the letters RY from Rosema, NE from Pauli, and LA from Lei.
  Fact 1.4: Someone eventually added an extra L to the resulting word.
Sentence 2:  They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at nearby Avery Island.
  Fact 2.0: The three women were daughters of Edward Avery McIlhenny.
  Fact 2.1: Edward Avery McIlhenny was a noted Louisiana conservationist.
  Fact 2.2: Edward Avery McIlhenny presided over McIlhenny Company.
  Fact 2.3: McIlhenny Company is the maker of Tabasco brand pepper sauce.
  Fact 2.4: McIlhenny Company is at nearby Avery Island.
Sentence 3: Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.
  Fact 3.0: Tabasco sauce is a brand of hot sauce.
  Fact 3.1: Tabasco sauce is made exclusively from tabasco peppers.
  Fact 3.2: The tabasco pepper is Capsicum frutescens var. tabasco.
Sentence 4:  "tabasco"), vinegar, and salt.
  Fact 4.0: Tabasco sauce is made with vinegar.
  Fact 4.1: Tabasco sauce is made with salt.

Retrieved items:
<item 1>
Rynella, Louisiana
Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway
329 (also known as the Avery Island Highway).

 The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA"
McIlhenny (someone eventually adding an extra L to the resulting word).

 They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at
nearby Avery Island.

 (At one time McIlhenny operated a general store at Rynella and he owned much of the land that now comprises and surrounds the community.)
</item 1>

<item 2>
Tabasco sauce
Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.

 "tabasco"), vinegar, and salt.

 It is produced by McIlhenny Company of Avery Island, Louisiana.
</item 2>

<item 3>
Louisiana Hot Sauce
Louisiana Hot Sauce is a brand of hot sauce manufactured in New Iberia, Louisiana by The Original Louisiana Hot Sauce Co., which is owned by Southeastern Mills
Inc.

 The product's labeling includes the word "original", and it is sometimes referred to as "Original Louisiana Hot Sauce" and "Original Louisiana Brand Hot
Sauce."

 It is a common hot sauce in the U.S. state of Louisiana.

 Bruce Foods was the previous owner and manufacturer of the brand, and sold it to Southeastern Mills Inc. in April 2015.
</item 3>

<item 4>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 4>

<item 5>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 5>

<item 6>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 6>

<item 7>
Crystal Hot Sauce
Crystal Hot Sauce is a brand of Louisiana-style hot sauce produced by family-owned Baumer Foods since 1923.

 3 e6USgal of Crystal Hot Sauce are shipped per year to 75 countries.

 The sauce is reddish orange with a medium heat and a milder, brighter flavor than Tabasco sauce.
</item 7>

<item 8>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 8>

<item 9>
Valentina (hot sauce)
Valentina is a brand of pourable hot sauce manufactured by Salsa Tamazula, a company in Guadalajara, Mexico.

 It is typically sold in 12.5-ounce and very large (one-liter or 34-ounce) glass bottles, with a flip-top cap permanently attached to the bottle (the cap does
not unscrew).

 The sauce, and the parent company's Tamazula hot sauce, are made with puya chilis from Jalisco state – similar to the Guajillo chili and known by the name
"guajillo puya" – is described as thicker than Tabasco sauce and less vinegary, with more chile flavor.

 Valentina comes in two varieties, hot and extra hot, and the sauce is known for its use on tacos and its taste, as opposed to only for its heat.

 Valentina's ingredients are water, chili peppers, vinegar, salt, spices and sodium benzoate (as a preservative).
</item 9>

<item 10>
Sriracha sauce
Sriracha (Thai: ศรีราชา ,  ] ; ) is a type of hot sauce or chili sauce made from a paste of chili peppers, distilled vinegar, garlic, sugar, and salt.

 It is named after the coastal city of Si Racha, in Chonburi Province of eastern Thailand, where it may have been first produced for dishes served at local
seafood restaurants.
</item 10>
=== END ITEM id=b71f9616ab2f99ac ===

=== ITEM id=8a33f1866dbe130a ===
Gold sentences and facts:
Sentence 0: The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US skyscraper.
  Fact 0.0: The Woolworth Building is at 233 Broadway, Manhattan, New York City.
  Fact 0.1: The Woolworth Building was designed by architect Cass Gilbert.
  Fact 0.2: The Woolworth Building was constructed between 1910 and 1912.
  Fact 0.3: The Woolworth Building is an early US skyscraper.
Sentence 1: 1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.
  Fact 1.0: 1 New York Plaza is an office building in New York City's Financial District.
  Fact 1.1: 1 New York Plaza was built in 1969.
  Fact 1.2: 1 New York Plaza is at the intersection of South and Whitehall Streets.

Retrieved items:
<item 1>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 1>

<item 2>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 2>

<item 3>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 3>

<item 4>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 4>

<item 5>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 5>

<item 6>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 6>

<item 7>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 7>

<item 8>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 8>

<item 9>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 9>

<item 10>
F. W. Woolworth Building (Watertown, New York)
The Woolworth Building is an historic building in Watertown, New York.

 It is a contributing building in the Public Square Historic District.

 Plans for the Woolworth Building were begun in 1916 by Frank W. Woolworth, the founder of the Woolworth's chain of department stores.
</item 10>
=== END ITEM id=8a33f1866dbe130a ===

=== ITEM id=0a4051b8d9ffb964 ===
Gold sentences and facts:
Sentence 0: Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.
  Fact 0.0: Pietro Alessandro Gaspare Scarlatti was born on 2 May 1660.
  Fact 0.1: Pietro Alessandro Gaspare Scarlatti died on 22 October 1725.
  Fact 0.2: Pietro Alessandro Gaspare Scarlatti was an Italian Baroque composer.
  Fact 0.3: Pietro Alessandro Gaspare Scarlatti was especially famous for his operas.
  Fact 0.4: Pietro Alessandro Gaspare Scarlatti was especially famous for his chamber cantatas.
Sentence 1: Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky, was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.
  Fact 1.0: The Russian name of Pyotr Ilyich Tchaikovsky is Пётр Ильи́ч Чайко́вский.
  Fact 1.1: Pyotr Ilyich Tchaikovsky was born on 25 April/7 May 1840.
  Fact 1.2: Pyotr Ilyich Tchaikovsky died on 25 October/6 November 1893.
  Fact 1.3: Pyotr Ilyich Tchaikovsky is often anglicized as Peter Ilich Tchaikovsky.
  Fact 1.4: Pyotr Ilyich Tchaikovsky was a Russian composer of the romantic period.
  Fact 1.5: Some works by Pyotr Ilyich Tchaikovsky are among the most popular music in the classical repertoire.
Sentence 2:  He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the United States.
  Fact 2.0: Tchaikovsky was the first Russian composer whose music made a lasting impression internationally.
  Fact 2.1: The international impression of Tchaikovsky's music was bolstered by his appearances as a guest conductor.
  Fact 2.2: Tchaikovsky appeared as a guest conductor in Europe.
  Fact 2.3: Tchaikovsky appeared as a guest conductor in the United States.

Retrieved items:
<item 1>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 1>

<item 2>
Piano Sonata in C-sharp minor (Tchaikovsky)
The Piano Sonata in C-sharp minor, Op.

 posth.

 80, was written by Russian composer Pyotr Ilyich Tchaikovsky in 1865, his last year as a student at the St Petersburg Conservatory.

 The sonata in its original form was not published in Tchaikovsky's lifetime; it was published in 1900 by P. Jurgenson, and given the posthumous opus number 80.
</item 2>

<item 3>
International Tchaikovsky Competition
The International Tchaikovsky Competition is a classical-music competition held every four years in Moscow, Russia, for pianists, violinists, and cellists
between 16 and 32 years of age, and singers between 19 and 32 years of age.

 The competition is named after Russian composer Pyotr Ilyich Tchaikovsky and is an active member of the World Federation of International Music Competitions.
</item 3>

<item 4>
Tchaikovsky (film)
Tchaikovsky (Russian: Чайковский ) is a 1970 Soviet biopic film directed by Igor Talankin.

 It featured Innokenty Smoktunovsky in the role of the famous Russian composer Pyotr Ilyich Tchaikovsky.

 It was nominated for the 1971 Academy Award for Best Foreign Language Film as well as the Academy Award for Original Song Score and Adaptation.
</item 4>

<item 5>
Chaikovskij (crater)
Chaikovskij (sometimes Tchaikovsky) is a crater on Mercury.

 It has a diameter of 165 kilometers.

 Its name was adopted by the International Astronomical Union in 1976.

 Chaikovskij is named for the Russian composer Pyotr Ilyich Tchaikovsky, who lived from 1840 to 1893.
</item 5>

<item 6>
Kamarinskaya
Kamarinskaya (Russian: камаринская ) is a Russian traditional folk dance, which is mostly known today as the Russian composer Mikhail Glinka's composition of
the same name.

 Glinka's "Kamarinskaya", written in 1848, was the first orchestral work based entirely on Russian folk song and to use the compositional principles of that
genre to dictate the form of the music.

 It became a touchstone for the following generation of Russian composers ranging from the Western-oriented Pyotr Ilyich Tchaikovsky to the group of
nationalists known collectively as The Five and was also lauded abroad, most notably by French composer Hector Berlioz.
</item 6>

<item 7>
Piano Concerto No. 2 (Tchaikovsky)
Pyotr Ilyich Tchaikovsky's Piano Concerto No. 2 in G major, Op.

 44, was written in 1879–1880.

 It was dedicated to Nikolai Rubinstein, who had insisted he be allowed to perform it at the premiere as a way of making up for his harsh criticism of
Tchaikovsky's First Piano Concerto.

 Rubinstein was destined never to play it, however, as he died in March 1881.

 The premiere performance took place in New York City, on 12 November 1881.

 The soloist was Madeline Schiller, and Theodore Thomas conducted the New York Philharmonic orchestra.

 The first Russian performance was in Moscow in May 1882, conducted by Anton Rubinstein with Tchaikovsky's pupil, Sergei Taneyev, at the piano.
</item 7>

<item 8>
Symphony No. 2 (Rachmaninoff)
Symphony No. 2 in E minor, Op. 27 is a symphony by the Russian composer Sergei Rachmaninoff, written in 1906–07.

 The premiere was conducted by the composer himself in Saint Petersburg on 8 February 1908.

 Its duration is approximately 60 minutes when performed uncut; cut performances can be as short as 35 minutes.

 The score is dedicated to Sergei Taneyev, a Russian composer, teacher, theorist, author, and pupil of Pyotr Ilyich Tchaikovsky.

 Alongside his Piano Concerto No. 2 and Piano Concerto No. 3, this symphony remains one of the composer's best known compositions.
</item 8>

<item 9>
Pyotr Ilyich Tchaikovsky
Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky,
was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.

 He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the
United States.

 Tchaikovsky was honored in 1884, by Emperor Alexander III, and awarded a lifetime pension.
</item 9>

<item 10>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 10>
=== END ITEM id=0a4051b8d9ffb964 ===

=== ITEM id=2689a5bedc1f0710 ===
Gold sentences and facts:
Sentence 0: ", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)
  Fact 0.0: This person did voice work for My Life As A Teenage Robot.
  Fact 0.1: This person did voice work for The Replacements.
  Fact 0.2: This person did the voice of Knotty in a 1992 work whose title is blank in the sentence.
  Fact 0.3: This person did additional voices in the film Fantasia 2000 (1999).
  Fact 0.4: This person did additional voices in the film Sing (2016).
Sentence 1:  It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.
  Fact 1.0: The film was directed by Garth Jennings.
  Fact 1.1: The film was written by Garth Jennings.
  Fact 1.2: The film was co-directed by Christophe Lourdelet.
  Fact 1.3: The film starred the voices of Matthew McConaughey.
  Fact 1.4: The film starred the voices of Reese Witherspoon.
  Fact 1.5: The film starred the voices of Seth MacFarlane.
  Fact 1.6: The film starred the voices of Scarlett Johansson.
  Fact 1.7: The film starred the voices of John C. Reilly.
  Fact 1.8: The film starred the voices of Taron Egerton.
  Fact 1.9: The film starred the voices of Tori Kelly.

Retrieved items:
<item 1>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 1>

<item 2>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 2>

<item 3>
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 3>

<item 4>
Are We Done Yet?
Are We Done Yet?

 is a 2007 American family comedy film starring Ice Cube.

 The film is a remake of the Cary Grant comedy "Mr. Blandings Builds His Dream House" and a sequel to the 2005 comedy "Are We There Yet?

" The film was directed by Steve Carr from a screenplay by Hank Nelken.

 It was produced by Revolution Studios and RKO Pictures and distributed by Columbia Pictures.

 The film was shot on location in Tsawwassen, British Columbia, Canada, but is set in Newberg, Oregon, United States.
</item 4>

<item 5>
Winnie the Pooh (2011 film)
Winnie the Pooh is a 2011 American animated buddy musical comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 51st Disney animated feature film.

 Inspired by A. A. Milne's stories of the same name, the film is part of Disney's "Winnie the Pooh" franchise, the fifth theatrical "Winnie the Pooh" film
released, and Walt Disney Animation Studios' second adaptation of "Winnie-the-Pooh" stories.

 Jim Cummings reprises his vocal roles as Winnie the Pooh and Tigger, while series newcomers Travis Oates, Tom Kenny, Craig Ferguson, Bud Luckey, and Kristen
Anderson-Lopez provide the voices of Piglet, Rabbit, Owl, Eeyore, and Kanga, respectively.

 In the film, the aforementioned residents of the Hundred Acre Wood embark on a quest to save Christopher Robin from an imaginary culprit while Pooh deals with
a hunger for honey.
</item 5>

<item 6>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
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
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 8>

<item 9>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 9>

<item 10>
Small Apartments
Small Apartments is a 2012 American comedy film directed by Jonas Åkerlund.

 It tells the story of Franklin Franklin, played by Matt Lucas, who by mistake kills his landlord, Mr. Olivetti, played by Peter Stormare.

 The cast co-stars Dolph Lundgren, Johnny Knoxville, James Caan, Billy Crystal, Juno Temple, Rebel Wilson, Saffron Burrows and Amanda Plummer.

 The screenplay was written by Chris Millis and adapted from his own novella.

 The film premiered at the South by Southwest Film Festival on March 10, 2012.
</item 10>
=== END ITEM id=2689a5bedc1f0710 ===

=== ITEM id=a5d4b56c7742d216 ===
Gold sentences and facts:
Sentence 0:  It is found in Yemen and India.
  Fact 0.0: The subject of this sentence is found in Yemen.
  Fact 0.1: The subject of this sentence is found in India.
Sentence 1:  Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.
  Fact 1.0: Yemen's capital has been temporarily relocated to the port city of Aden.
  Fact 1.1: Aden is on the southern coast.
  Fact 1.2: The relocation is stated to be because of the content of the previous sentence.

Retrieved items:
<item 1>
Aulacodes templalis
Aulacodes templalis is a moth in the Crambidae family.

 It was described by Schaus in 1906.

 It is found in Brazil.
</item 1>

<item 2>
Ornithodoros savignyi
Because of its habit of feeding and dropping from its host, adult dispersal is limited, whereas larvae may remain attached to their hosts for several days.

 During its life cycle it will feed on multiple hosts between moults.
</item 2>

<item 3>
Al Khor (city)
Al Khor is a coastal city in northern Qatar, located 50 kilometres north of the capital, Doha.

 It is the capital city of the municipality of Al Khor.

 The name of the city means creek in Arabic as the town is located on a creek.

 Al Khor is home to many employees of the oil industry due to its proximity to Qatar's northern oil and natural gas fields, and due to its proximity to the Ras
Laffan Industrial City.
</item 3>

<item 4>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 4>

<item 5>
Amka
Amka (Hebrew: עַמְקָה‎ ), also known in Arabic as Amqa (Arabic: عمقا‎ ‎ ), is a moshav in the Matte Asher Regional Council of Israel's Northern District, near
Acre.

 The location of the moshav roughly corresponds the former Arab village, depopulated during the 1948 Arab–Israeli War.

 Yemenite Jews, who arrived from the southern Arab country of Yemen, founded the village's successor Amka in 1949.

 In 2016 its population was 706 .
</item 5>

<item 6>
Agrotis orthogonia
The Pale Western Cutworm ("Agrotis orthogonia") is a moth of the Noctuidae family.

 It is found in North America, more specifically dry, semi-desert areas of western North America from southern Canada to California, ranging eastward nearly to
the eastern edge of the Great Plains.
</item 6>

<item 7>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 7>

<item 8>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 8>

<item 9>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 9>

<item 10>
Yemen
Yemen ( ; Arabic: اليَمَن‎ ‎ "al-Yaman "), officially known as the Republic of Yemen (الجمهورية اليمنية "al-Jumhūrīyah al-Yamanīyah "), is an Arab country in
Western Asia at the southern end of the Arabian Peninsula.

 Yemen is the second-largest country in the peninsula, occupying 527,970 km (203,850 sq mi).

 The coastline stretches for about 2,000 km (1,200 mi).

 It is bordered by Saudi Arabia to the north, the Red Sea to the west, the Gulf of Aden and Arabian Sea to the south, and Oman to the east-northeast.

 Although Yemen's constitutionally stated capital is the city of Sana'a, the city has been under rebel control since February 2015.

 Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.

 Yemen's territory includes more than 200 islands; the largest of these is Socotra.
</item 10>
=== END ITEM id=a5d4b56c7742d216 ===

=== ITEM id=6bc312bc73fd020e ===
Gold sentences and facts:
Sentence 0: Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.
  Fact 0.0: Orlando Brown was born on December 4, 1987.
  Fact 0.1: Orlando Brown is an American actor.
  Fact 0.2: Orlando Brown is a voice actor.
  Fact 0.3: Orlando Brown is a rapper.
  Fact 0.4: Orlando Brown is a singer.
Sentence 1:  He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
  Fact 1.0: Orlando Brown is best known for his role as Eddie Thomas in "That's So Raven".
  Fact 1.1: Orlando Brown is best known for his role as 3J Winslow in "Family Matters".
  Fact 1.2: Orlando Brown is best known for his role as Tiger in "Major Payne".
  Fact 1.3: Orlando Brown is best known for his role as Max in "Two of a Kind".
  Fact 1.4: Orlando Brown is best known for his role as Damey Wayne in the short-lived Waynehead.
  Fact 1.5: Orlando Brown is best known for his role as Dobbs in "Max Keeble's Big Move".
  Fact 1.6: Orlando Brown is best known for his role as Frankie in "Eddie's Million Dollar Cook Off".
Sentence 2: Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17, 1998.
  Fact 2.0: Family Matters is an American sitcom.
  Fact 2.1: Family Matters originated on ABC.
  Fact 2.2: Family Matters originated on ABC from September 22, 1989 to May 9, 1997.
  Fact 2.3: Family Matters moved to CBS after its ABC run.
  Fact 2.4: Family Matters was on CBS from September 19, 1997 to July 17, 1998.
Sentence 3:  A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.
  Fact 3.0: Family Matters is a spin-off of "Perfect Strangers."
  Fact 3.1: Family Matters revolves around the Winslow family.
  Fact 3.2: The Winslow family is a middle-class African American family.
  Fact 3.3: The Winslow family lives in Chicago, Illinois.

Retrieved items:
<item 1>
Darius McCrary
Darius Creston McCrary (born May 1, 1976) is an American film and television actor and singer.

 He is best known for his role as Edward "Eddie" Winslow, the oldest child of Carl and Harriette Winslow on the ABC/CBS television sitcom "Family Matters "which
ran from 1989–1998.

 Another one of McCrary's notable roles was as Scam in the 1987 comedy film "Big Shots", which was his film debut.

 He provided the voice of Jazz in the 2007 "Transformers" reboot.

 From December 2009 to October 2011, he portrayed photographer Malcolm Winters on the CBS daytime drama "The Young and the Restless".
</item 1>

<item 2>
Orlando Brown (actor)
Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.

 He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne
in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
</item 2>

<item 3>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 3>

<item 4>
In Living Color
In Living Color is an American sketch comedy television series that originally ran on Fox from April 15, 1990, to May 19, 1994.

 Brothers Keenen and Damon Wayans created, wrote and starred in the program.

 The show was produced by Ivory Way Productions in association with 20th Century Fox Television and was taped at stage 7 at the Fox Television Center on Sunset
Boulevard in Hollywood, California.

 The title of the series was inspired by the NBC announcement of broadcasts being presented "in living color" during the 1960s, prior to mainstream color
television.

 It also refers to the fact that most of the show's cast were black, unlike other sketch comedy shows such as "Saturday Night Live" whose casts are mostly
white.

 It was controversial due to the Wayans' decision to portray African-American humor from the ghetto in a time when mainstream American tastes regarding black
comedy had been set by more upscale shows such as "The Cosby Show", causing an eventual feud for control between Fox executives and the Wayans.
</item 4>

<item 5>
Lamman Rucker
Lamman Rucker (born October 6, 1971) is an American actor.

 Rucker began his career on the daytime soap operas "As the World Turns" and "All My Children", before roles in Tyler Perry's films "Why Did I Get Married?

", "Why Did I Get Married Too?

", and "Meet the Browns", and its television adaptation.

 In 2016, he began starring as Jacob Greenleaf in the Oprah Winfrey Network drama series, "Greenleaf".
</item 5>

<item 6>
Randy J. Goodwin
Randy J. Goodwin, born as Ivan Jerome Goodwin, is a Hollywood actor.

 He is perhaps best known for his role as "Kennedy Winslow" on "Fast Track" and "Max Ellis" on the show "Abby", as well as his recurring roles on "Girlfriends"
as "Davis Hamilton" and "The Vampire Diaries" as "Dr. Jonas Martin".
</item 6>

<item 7>
Jaimee Foxworth
Jaimee Monae Foxworth (born December 17, 1979) is an American actress and model.

 She is best known for her role of Judy Winslow, the youngest daughter of Carl and Harriette Winslow on the ABC/CBS sitcom "Family Matters".

 She later transitioned to pornographic films using the name Crave, starring in several titles.
</item 7>

<item 8>
Family Matters
Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17,
1998.

 A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.

 Midway through the first season, the show introduced the Winslows' nerdy neighbor Steve Urkel (Jaleel White), who quickly became its breakout character and
eventually the show's main character.

 Having run for nine seasons, "Family Matters" became the second longest-running non-animated US sitcom with a predominantly African American cast, behind only
"The Jeffersons" (11).

 Having aired 215 episodes, "Family Matters" is ranked third, behind only "Tyler Perry's House of Payne" (254), and "The Jeffersons" (253).
</item 8>

<item 9>
Michael Jai White
Michael Jai White (born November 10, 1967) is an American actor and martial artist who has appeared in numerous films and television series.

 He is the first African American to portray a major comic book superhero in a major motion picture, having starred as Al Simmons, the protagonist in the 1997
film "Spawn".

 White appeared as Marcus Williams in the Tyler Perry films "Why Did I Get Married?

" and "Why Did I Get Married Too?

", and starred as the character on the TBS/OWN comedy-drama television series "Tyler Perry's For Better or Worse".

 White portrayed Jax Briggs in "".

 White also portrayed boxer Mike Tyson in the 1995 HBO television movie "Tyson", and starred as Black Dynamite.
</item 9>

<item 10>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 10>
=== END ITEM id=6bc312bc73fd020e ===

=== ITEM id=0b979e58a66e9bdf ===
Gold sentences and facts:
Sentence 0: Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.
  Fact 0.0: Talking Heads were an American rock band.
  Fact 0.1: Talking Heads were formed in 1975.
  Fact 0.2: Talking Heads were formed in New York City.
  Fact 0.3: Talking Heads were active until 1991.
Sentence 1:  is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
  Fact 1.0: The group is a surf rock group.
  Fact 1.1: The group formed in Auburn, Alabama.
  Fact 1.2: The group formed in the early 1990s.
  Fact 1.3: The group came to prominence over the following decade.

Retrieved items:
<item 1>
New World Man
"New World Man" is a track from the 1982 album "Signals" by Canadian rock band Rush.

 The song was the last and quickest composed song on the album, stemming from a suggestion by then-Rush producer Terry Brown to even out the lengths of the two
sides of the cassette version.

 It went to #1 (on the "RPM" national singles chart) in Canada, where it remained for two weeks in October 1982.

 Less successful in the United States, it nonetheless remains Rush's only American Top 40 hit, peaking at #21 on the "Billboard" singles chart for three weeks
in October and November 1982.

 It also topped the "Billboard" Top Tracks chart for two weeks (their first single to do so).

 "New World Man" also reached #42 in the UK; a remixed version released as a double A-side with "Countdown" later reached #36 in the UK in early 1983.
</item 1>

<item 2>
Talking Heads
Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.

 The band comprised David Byrne (lead vocals, guitar), Chris Frantz (drums), Tina Weymouth (bass), and Jerry Harrison (keyboards, guitar).

 Described by critic Stephen Thomas Erlewine as "one of the most critically acclaimed bands of the '80s," the group helped to pioneer new wave music by
integrating elements of punk, art rock, funk, and world music with avant-garde sensibilities and an anxious, clean-cut image.
</item 2>

<item 3>
No Talking, Just Head
No Talking, Just Head is an album released in 1996 by The Heads, a band composed of Jerry Harrison, Tina Weymouth, and Chris Frantz of Talking Heads, joined by
a variety of guest singers.

 Its name may be seen as an allusion to the fact that Talking Heads' former vocalist, David Byrne, is the only member not involved.
</item 3>

<item 4>
More Songs About Buildings and Food
More Songs About Buildings and Food is the second studio album by the American rock band Talking Heads, released in July 1978.

 The album was the first of three Talking Heads LPs produced by collaborator Brian Eno.

 It saw the group move musically toward a danceable style, crossing singer David Byrne's unusual delivery with new emphasis on the rhythm section (made up of
bassist Tina Weymouth and drummer Chris Frantz).
</item 4>

<item 5>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 5>

<item 6>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 6>

<item 7>
Head Automatica
Head Automatica is an American rock band from Long Island, New York fronted by Daryl Palumbo (also of Glassjaw).
</item 7>

<item 8>
Method Man
Clifford Smith (born April 1, 1971), better known by his stage name Method Man, is an American rapper, record producer, and actor.

 He is known as a member of the East Coast hip hop collective Wu-Tang Clan.

 He is also one half of the hip hop duo Method Man & Redman.

 He took his stage name from the 1979 film "Method Man".

 In 1996, he won a Grammy Award for Best Rap Performance by a Duo or Group, for "I'll Be There for You/You're All I Need to Get By", with American R&B
singer-songwriter Mary J. Blige.
</item 8>

<item 9>
Talking Heads Africa
Talking Heads (Africa) was introduced in Cape Town in 2008 as part of the Infecting the City public art festival.

 Talking Heads has four core components that form the project.

 These include: developing a platform for conversation and exchange with and between experts; creating a network of African thought leaders; shooting
mini-documentaries that define these leaders and their contributions; developing the tools to make this model work in cities all over the African continent.
</item 9>

<item 10>
Man or Astro-man?
Man or Astro-man?

 is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
</item 10>
=== END ITEM id=0b979e58a66e9bdf ===

=== ITEM id=d02bbd8de85ec667 ===
Gold sentences and facts:
Sentence 0:  For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record when her album "25" took just 29 days to reach the landmark.
  Fact 0.0: For almost 20 years, a record (held by a subject not named in this sentence) stood for selling 2 million copies in the UK in the fastest time.
  Fact 0.1: The record for selling 2 million copies in the UK in the fastest time was set in 42 days.
  Fact 0.2: In 2015, Adele surpassed the record.
  Fact 0.3: Adele's album '25' took just 29 days to reach the 2 million copies landmark.
Sentence 1:  At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
  Fact 1.0: The 51st Grammy Awards were held in 2009.
  Fact 1.1: Adele received the Best New Artist award at the 51st Grammy Awards.
  Fact 1.2: Adele received the Best Female Pop Vocal Performance award at the 51st Grammy Awards.

Retrieved items:
<item 1>
List of Billboard 200 number-one albums of 2002
Country music singer Shania Twain's album "Up!

" opened up with a huge first week sales of 857,000 copies in its first week alone, giving her the recognition of the highest first week sales of her career and
second highest of the year, only behind Eminem's "The Eminem Show" and at the time the fastest selling solo female album ever.

 Nelly's album "Nellyville" opened up with his highest first week sales of his career which logged on with huge sales of 714,000 copies in its first week alone,
which beat his sales of his debut album "Country Grammar", which opened up with first week sales of 235,000 copies.

 Country singer Alan Jackson album "Drive" gave him his first number one album on the chart and opened up with first week sales of 211,000 copies alone.
</item 1>

<item 2>
List of Billboard 200 number-one albums of 2002
The band Creed continued its eight week long run on the chart but is credited as the longest running album 2001.

 Jennifer Lopez earned her second number one album on the charts with "", which became the highest first week sales of a remix album at the time.

 R&B artist Ashanti earned her first number one album with her self-titled debut album "Ashanti", which opened up with first week sales of 503,000 copies in its
first week alone.

 Puff Daddy earned his first number one album since "No Way Out" back in 1997.

 Rapper Jay-Z earned his fifth chart topper with "", which opened up with first week sales of 545,000 copies alone.

 Heavy metal band Disturbed earned its first number one album on the chart with "Believe", which opened up with first week sales of 284,000 copies alone.

 Country music singer Shania Twain's album "Up!

"
</item 2>

<item 3>
MTV Video Music Award for Best New Artist
The MTV Video Music Award for Best New Artist have been given out since the first annual MTV Video Music Awards in 1984.

 From then to 2006, the award was named Best New Artist in a Video.

 In 2007 its name was changed to Best New Artist, as the category underwent a format change to award the artist's body of work for the full year rather than a
specific video.

 For the 2008 ceremony, though, while the award retained its 2007 name, it returned to the format of awarding a specific video rather than the artist's full
body of work.

 The category was later renamed Artist to Watch through 2013 to 2015 while still keeping the format of an award going to a certain video.
</item 3>

<item 4>
Billboard Year-End
"Billboard" Year-End charts are a cumulative measure of a single or album's performance in the United States, based upon the "Billboard" magazine charts during
any given chart year.

 "Billboard's" "chart year" runs from the first Billboard "week" of December to the final week in November, but because the Billboard week is dated in advance
of publication, the last calendar week for which sales are counted is usually the third week in November.

 This altered calendar allows for "Billboard" to calculate year-end charts and release them in time for its final print issue in the last week of December.

 Prior to incorporating chart data from Nielsen SoundScan (from 1991), year-end charts were calculated by an inverse-point system based solely on a title's
performance (for example a single appearing on the "Billboard" Hot 100 would be given one point for a week spent at position 100, two points for a week spent at
position ninety-nine, and so forth, up to 100 points for each week spent at number one).

 Other factors including the total weeks a song spent on the chart and at its peak position were calculated into its year-end total.
</item 4>

<item 5>
Don't Stop the Music (Robyn album)
Don't Stop the Music is the third studio album by Swedish pop singer Robyn.

 It was released on 30 October 2002 in Sweden by BMG.

 The album peaked at #2 in her native Sweden, and the two singles "Keep This Fire Burning" and "Don't Stop the Music" were both top 10 hits.

 In 2003, "Don't Stop The Music" was certified platinum by IFPI, and has sold over 60,000 copies in Sweden.
</item 5>

<item 6>
List of accolades received by Grand Theft Auto V
Within twenty-four hours of its release, "Grand Theft Auto V" generated more than $815.7 million in worldwide revenue, equating to approximately 11.21 million
copies sold for Take Two.

 Three days after release, the game had surpassed one billion dollars in sales, making it the fastest selling entertainment product in history.
</item 6>

<item 7>
List of Billboard 200 number-one albums of 2002
The highest-selling albums and EPs in the United States are ranked in the "Billboard" 200, published by "Billboard" magazine.

 The data are compiled by Nielsen Soundscan based on each album's weekly physical and digital sales.

 25 acts achieved number one albums during this year with artist such as Nelly and Shania Twain who had their albums debut at number one on the chart.

 Rapper Eminem's "The Eminem Show" is the best selling album of 2002 selling over approximately 7.6 million copies by the end of the year.

 It is also the longest running album of 2002 spending six non-consecutive weeks the chart and was known for its first full week of sales debut of 1.322 million
copies which Nielsen SoundScan scanned as the sixth largest sales of all time in its first week.

 Its debut of 1.322 million copies has still not been matched by any album today since except for Taylor Swift's album "1989", which opened with first week
sales of 1.279 million copies.
</item 7>

<item 8>
Adele
Adele Laurie Blue Adkins (born 5 May 1988) ( ) is an English singer-songwriter.

 After graduating from the BRIT School for Performing Arts and Technology in 2006, Adele was given a recording contract by XL Recordings after a friend posted
her demo on Myspace the same year.

 In 2007, she received the Brit Awards "Critics' Choice" award and won the BBC Sound of 2008 poll.

 Her debut album, "19", was released in 2008 to commercial and critical success.

 It is certified seven times platinum in the UK, and three times platinum in the US.

 The album contains her first song, "Hometown Glory", written when she was 16, which is based on her home suburb of West Norwood in London.

 An appearance she made on "Saturday Night Live" in late 2008 boosted her career in the US.

 At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
</item 8>

<item 9>
Robson &amp; Jerome (album)
Robson & Jerome is the debut studio album by British singing duo Robson & Jerome.

 It was released in the UK in 1995 by RCA Records, and reached number one on the UK Albums Chart.

 It was the Christmas number one album of 1995, and was the biggest-selling album of the year, with sales of 2,040,000.

 For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record
when her album "25" took just 29 days to reach the landmark.
</item 9>

<item 10>
Billboard Year-End
The same method was used for albums based on the "Billboard" 200, and songs appearing on the other charts (e.g. Hot Country Singles).
</item 10>
=== END ITEM id=d02bbd8de85ec667 ===
