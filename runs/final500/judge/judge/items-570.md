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

=== ITEM id=bd586dd0a6a71e7d ===
Gold sentences and facts:
Sentence 0: Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the electronica band Republica.
  Fact 0.0: Samantha Marie Sprackling was born on 3 June 1968.
  Fact 0.1: Samantha Marie Sprackling was born in Lagos, Nigeria.
  Fact 0.2: Samantha Marie Sprackling is more commonly known as Saffron.
  Fact 0.3: Samantha Marie Sprackling is a singer.
  Fact 0.4: Samantha Marie Sprackling is a songwriter.
  Fact 0.5: Samantha Marie Sprackling is an actress.
  Fact 0.6: Samantha Marie Sprackling is the lead singer of the electronica band Republica.
Sentence 1: Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.
  Fact 1.0: Myles Richard Kennedy was born Myles Richard Bass.
  Fact 1.1: Myles Richard Kennedy was born on November 27, 1969.
  Fact 1.2: Myles Richard Kennedy is an American musician.
  Fact 1.3: Myles Richard Kennedy is an American singer.
  Fact 1.4: Myles Richard Kennedy is an American songwriter.
  Fact 1.5: Myles Richard Kennedy is best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge.
  Fact 1.6: Myles Richard Kennedy is the lead vocalist in guitarist Slash's backing band.
  Fact 1.7: Slash's backing band is known as Myles Kennedy and the Conspirators.
Sentence 2:  A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several artists, and has been involved with several projects throughout his career.
  Fact 2.0: Myles Richard Kennedy is a former guitar instructor.
  Fact 2.1: Myles Richard Kennedy is from Spokane, Washington.
  Fact 2.2: Myles Richard Kennedy has worked as a session musician.
  Fact 2.3: Myles Richard Kennedy has worked as a songwriter.
  Fact 2.4: Myles Richard Kennedy has made studio appearances with several artists.
  Fact 2.5: Myles Richard Kennedy has made live appearances with several artists.
  Fact 2.6: Myles Richard Kennedy has been involved with several projects throughout his career.

Retrieved items:
<item 1>
World on Fire (Slash song)
"World on Fire" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it is the title track of the guitarist's third solo album (the second with Kennedy and The Conspirators) "World on Fire".

 Released as the album's lead single, "World on Fire" topped the US "Billboard" Mainstream Rock Songs chart.
</item 1>

<item 2>
List of songs recorded by Myles Kennedy
Myles Kennedy is an American rock musician.

 Born in Boston, Massachusetts, he originally began his musical career in Spokane, Washington as the guitarist in jazz band Cosmic Dust.

 After recording 1991's "Journey", he left to form alternative rock band Citizen Swing in 1992.

 The group released two albums, 1992's "Cure Me with the Groove" and 1996's "Deep Down", on both of which Kennedy was co-credited for songwriting.

 Kennedy's next band was The Mayfield Four, which he co-founded with guitarist Craig Johnson, bassist Marty Meisner and drummer Zia Uddin in 1996.

 The frontman wrote most of the material on the band's 1998 debut "Fallout" and all of the songs on 2001's "Second Skin".
</item 2>

<item 3>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 3>

<item 4>
Apocalyptic Love World Tour
The Apocalyptic Love World Tour is the second concert tour by ex-Guns N' Roses lead guitarist Slash as a solo artist.

 The tour features the same backing band that toured with Slash during his first solo tour, though this time the act is billed as "Slash featuring Myles Kennedy
and The Conspirators".

 The tour is in support of Slash's second solo album "Apocalyptic Love", which features Myles Kennedy handling lead vocal duties on all songs, as well as his
touring band members: bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 4>

<item 5>
Myles Kennedy discography
The full discography of rock musician Myles Kennedy consists of eleven studio albums, two concert films, four live albums, two extended plays, and thirteen
singles in total, in addition to eleven studio tracks that he has appeared on as a featured artist, one of which was a single.

 Born in Boston on November 27, 1969, Kennedy is currently a member of the rock band Alter Bridge, with whom he has released four studio albums, two concert
films, and several singles.

 He is also the frontman of Slash's touring group, and with Slash he has released a live album, "Live in Manchester", the first of a series of live albums
released throughout the summer of 2010, and "", another live album released in 2011.

 In 2012, he released a collaboration studio album with Slash titled "Apocalyptic Love", which is billed to Slash featuring Myles Kennedy and the Conspirators,
as well as the 2014 followup titled "World on Fire".

 With The Mayfield Four, he released two studio albums, two extended plays, and four singles; with Citizen Swing, two studio albums; and with Cosmic Dust, one
studio album.
</item 5>

<item 6>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 6>

<item 7>
Myles Kennedy
Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm
guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.

 A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several
artists, and has been involved with several projects throughout his career.
</item 7>

<item 8>
Bent to Fly
"Bent to Fly" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it was released as the second single from the guitarist's third solo album (the second with Kennedy and The Conspirators), "World
on Fire".

 The song was used as the theme song for the 2014 National Rugby League Finals series, and Slash performed the song live at ANZ Stadium as pre-show
entertainment for the league's grand final that year.
</item 8>

<item 9>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 9>

<item 10>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 10>
=== END ITEM id=bd586dd0a6a71e7d ===

=== ITEM id=9a28b89815918aab ===
Gold sentences and facts:
Sentence 0: The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver Division, the Shenango Division, and the Conneaut Division.
  Fact 0.0: The Beaver and Erie Canal was also known as the Erie Extension Canal.
  Fact 0.1: The Beaver and Erie Canal was part of the Pennsylvania Canal system.
  Fact 0.2: The Beaver and Erie Canal consisted of three sections.
  Fact 0.3: The three sections of the Beaver and Erie Canal were the Beaver Division, the Shenango Division, and the Conneaut Division.
Sentence 1: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.
  Fact 1.0: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp.
  Fact 1.1: The Dismal Swamp Canal is located in Virginia and North Carolina.
  Fact 1.2: Virginia and North Carolina are in the United States.

Retrieved items:
<item 1>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 1>

<item 2>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
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
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 4>

<item 5>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 5>

<item 6>
Dismal Swamp State Park
Dismal Swamp State Park is a North Carolina state park in Camden County, North Carolina in the United States.

 The park was created as a state natural area in 1974 with the help of The Nature Conservancy, and on July 28, 2007 the NC General Assembly re-designated it as
a state park.

 It opened to the public in 2008.

 This marked the first time that public access to Great Dismal Swamp was made possible in North Carolina.

 The park covers 14432 acre of protected land on the North Carolina/Virginia border.

 Park offices are three miles (5 km) south of the border on U.S. Route 17 near South Mills.

 Features of the park include the canal which is used regularly by boaters using the Intracoastal Waterway and several miles of hiking and biking trails.
</item 6>

<item 7>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 7>

<item 8>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 8>

<item 9>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 9>

<item 10>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 10>
=== END ITEM id=9a28b89815918aab ===

=== ITEM id=33eb5c49a75a93ce ===
Gold sentences and facts:
Sentence 0: Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in 2011.
  Fact 0.0: Pramod Bhasin stepped down as President of Genpact.
  Fact 0.1: Pramod Bhasin stepped down as CEO of Genpact.
  Fact 0.2: Genpact is India's largest business process outsourcing (BPO) company.
  Fact 0.3: Pramod Bhasin became non-executive Vice Chairman in 2011.
Sentence 1: Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.
  Fact 1.0: Genpact is a global business process management and services corporation.
  Fact 1.1: Genpact is an IT corporation.
  Fact 1.2: Genpact has key offices in New York City, United States.

Retrieved items:
<item 1>
Profiles span companies, organizations, people, and technologies: Pharma Medica conducts pharmaceutical trials in Ontario and Missouri; Genpact is a global
BPM/IT firm, formerly led by Pramod Bhasin; Summit BHC operates U.S. addiction-treatment centers. Triple Canopy merged with Academi in 2014 to form Constellis.
Other subjects include geothermal binary-cycle power, FuelCell Energy, packaging maker Krones, Polish firms, media and investment groups, retailers,
</item 1>

<item 2>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 2>

<item 3>
Mike Long (American businessman)
and returning it to profitability.

 Touching on his experience of bringing about the initial public offerings of web-based businesses, Long would say that investors needed to be presented "with
an entirely new face every few months," and that "the only way to run one of these Silicon Valley companies was to forget everything you'd learned outside of
Silicon Valley."
</item 3>

<item 4>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 4>

<item 5>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 5>

<item 6>
Pramod Bhasin
Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in
2011.
</item 6>

<item 7>
Mike Long (American businessman)
Mike Long is an American business man, former CEO of several public companies, and currently a founding partner of Sulgrave Partners LLC.

 He served as the President and Chief Executive Officer of Continuum, an Austin, Texas IT consulting company, from 1991 to 1997, having started with Continuum
as a Director in 1983.

 In 1997, Long was named CEO of Healtheon Corporation (now WebMD), succeeding former CEO David Schnell.

 Long oversaw Healtheon's initial public offering, traveling between Europe and the United States to woo investors.

 Long was able to secure the required investment funds, and saw Healtheon's stock price rise from $8 to a high of $120.

 In 2002, Long was recruited to fix the financial struggles of Move, Inc., a company plagued by more than $4 billion in lawsuits and hemorrhaging tens of
millions of dollars a quarter.

 As Chief Executive Officer, Long was able to revive Homestore, Inc., by changing the business model, rebranding the company as Move, Inc. and returning it to
profitability.
</item 7>

<item 8>
Real estate benchmarking measures and compares property investments; proptech uses technology to improve real estate transactions. Bell Labs is Nokia-owned US
research company. Embase is Elsevier’s extensive biomedical and drug-literature database. Dental notation systems identify teeth; DUNS assigns unique business
identifiers. PIDBA compiles Paleoindian archaeological data. Dun & Bradstreet supplies business data and analytics. TBMA merged into SIFMA in 2006. A periplus
lists coastal ports and distances. BNB
</item 8>

<item 9>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 9>

<item 10>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 10>
=== END ITEM id=33eb5c49a75a93ce ===

=== ITEM id=ee172fb9d8c22453 ===
Gold sentences and facts:
Sentence 0: Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.
  Fact 0.0: Sleepless in Seattle is a 1993 film.
  Fact 0.1: Sleepless in Seattle is an American film.
  Fact 0.2: Sleepless in Seattle is a romantic comedy-drama film.
  Fact 0.3: Sleepless in Seattle was directed by Nora Ephron.
  Fact 0.4: Sleepless in Seattle was co-written by Nora Ephron.
  Fact 0.5: Sleepless in Seattle is based on a story by Jeff Arch.
Sentence 1:  It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber, and Rita Wilson.
  Fact 1.0: Sleepless in Seattle stars Tom Hanks.
  Fact 1.1: Sleepless in Seattle stars Meg Ryan.
  Fact 1.2: Sleepless in Seattle has a supporting cast featuring Bill Pullman.
  Fact 1.3: Sleepless in Seattle has a supporting cast featuring Ross Malinger.
  Fact 1.4: Sleepless in Seattle has a supporting cast featuring Rob Reiner.
  Fact 1.5: Sleepless in Seattle has a supporting cast featuring Rosie O'Donnell.
  Fact 1.6: Sleepless in Seattle has a supporting cast featuring Gaby Hoffmann.
  Fact 1.7: Sleepless in Seattle has a supporting cast featuring Victor Garber.
  Fact 1.8: Sleepless in Seattle has a supporting cast featuring Rita Wilson.
Sentence 2: Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.
  Fact 2.0: Roseann O'Donnell was born on March 21, 1962.
  Fact 2.1: Roseann O'Donnell is an American comedian.
  Fact 2.2: Roseann O'Donnell is an American actress.
  Fact 2.3: Roseann O'Donnell is an American author.
  Fact 2.4: Roseann O'Donnell is an American television personality.

Retrieved items:
<item 1>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 1>

<item 2>
Barry Sandler
Barry Sandler (born February 23, 1947 in Buffalo, New York) is an American screenwriter and film producer.

 His career has spanned several decades, with the 1980s being his most prolific.

 The openly gay Sandler is perhaps best known for writing the 1982 film "Making Love", the first mainstream Hollywood film to deal seriously with issues of
homosexuality and coming out.

 Sandler discussed "Making Love" in the 1995 documentary film "The Celluloid Closet".
</item 2>

<item 3>
The collection spans films and TV from 1947–2010: Bruce Willis’s dramas, action comedies and thrillers include *In Country* (Golden Globe-nominated), *Last Man
Standing* (a *Yojimbo* remake), *The Whole Nine Yards* and *Cop Out*. Other highlights: *The Bears and I*; teen comedies *Can’t Buy Me Love* and *The Night
Before*; *The Country Girl* (Oscar-winning screenplay
</item 3>

<item 4>
Reese Witherspoon (born 1976) is an American actress, producer, and entrepreneur who began as a child performer. Her films include *The Man in the Moon*,
*Pleasantville*, and *Election*, which earned her a Golden Globe nomination. Carrie Fisher (1956–2016) was an American actress, writer, and humorist best known
as Princess Leia in *Star Wars*. Her other credits include *The Blues Brothers* and *When Harry Met Sally
</item 4>

<item 5>
Carrie Fisher
Carrie Frances Fisher (October 21, 1956 – December 27, 2016) was an American actress, writer, and humorist.

 Fisher was known for playing Princess Leia in the "Star Wars" films; a role for which she was nominated for three Saturn Awards.

 Her other film roles included "Shampoo" (1975), "The Blues Brothers" (1980), "Hannah and Her Sisters" (1986), "The 'Burbs" (1989), "When Harry Met Sally..."
(1989), "Soapdish" (1991) and "The Women" (2008 film).

 She was nominated twice for the Primetime Emmy Award for Outstanding Guest Actress in a Comedy Series for her performances on the television series "30 Rock"
and "Catastrophe".

 She was posthumously made a Disney Legend in 2017.
</item 5>

<item 6>
In My Sleep
In My Sleep is a 2010 suspense thriller film written, directed and produced by film director Allen Wolf.

 It stars Philip Winchester, Lacey Chabert, Tim Draxl, Abigail Spencer and Kelly Overton.

 It also features Kirsten Vangsness from Criminal Minds in a cameo role and Tony Hale from Veep in a cameo as well.

 The story is about a massage therapist with chronic insomnia who fears he may have murdered a good friend while sleepwalking.
</item 6>

<item 7>
Matthew Perry
Matthew Langford Perry (born August 19, 1969) is a Canadian-American actor, best known for his role as Chandler Bing on the long-running NBC television sitcom
"Friends", as well as his portrayal of Ron Clark in the 2006 television movie "The Ron Clark Story".
</item 7>

<item 8>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 8>

<item 9>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 9>

<item 10>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 10>
=== END ITEM id=ee172fb9d8c22453 ===

=== ITEM id=666093c7677e9e8d ===
Gold sentences and facts:
Sentence 0:  In return, William made Copsi earl of Northumbria and sent him back to York.
  Fact 0.0: William made Copsi earl of Northumbria in return.
  Fact 0.1: William sent Copsi back to York.
Sentence 1: York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.
  Fact 1.0: York is a historic walled city.
  Fact 1.1: York is at the confluence of the rivers Ouse and Foss.
  Fact 1.2: York is in North Yorkshire, England.

Retrieved items:
<item 1>
Earl of Norbury
Earl of Norbury, in the County of Tipperary, is a title in the Peerage of Ireland.

 It was created in 1827, along with the title Viscount Glandine, of Glandine in the King's County, for the Irish politician and judge John Toler, 1st Baron
Norbury, upon his retirement as Chief Justice of the Common Pleas in Ireland.

 The titles were created with special remainder to his second son, Hector, as his eldest son, Daniel, was then considered mentally unwell.

 Lord Norbury had already been created Baron Norbury, of Ballycrenode in the County of Tipperary, in the Peerage of Ireland in 1800, with remainder to the heirs
male of his body.

 Moreover, his wife, Grace Toler (née Graham), had been created Baroness Norwood, of Knockalton in the County of Tipperary, in the Peerage of Ireland in 1797,
with remainder to the heirs male of her body.

 By the time Lord Norbury was raised to the Earldom, his wife had died and their eldest son had succeeded her as 2nd Baron Norwood.
</item 1>

<item 2>
Two Rivers (Red River of the North)
The Two Rivers is a 7.0 mi river in Kittson County, northwestern Minnesota, in the United States.

 Formed by the North Branch of the Two Rivers and the South Branch of the Two Rivers, it is a tributary of the Red River of the North, with its outflow
traveling north through Lake Winnipeg and the Nelson River to Hudson Bay.
</item 2>

<item 3>
Earl Cowper
He represented Hertford in Parliament.

 He was succeeded by his eldest son, the aforementioned William Cowper, the third Baronet, who was elevated to the peerage as Baron Cowper in 1706 and made Earl
Cowper in 1718.

 In 1706 Lord Cowper married as his second wife Mary Clavering, daughter of John Clavering, of Chopwell, County Durham.
</item 3>

<item 4>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 4>

<item 5>
Earl of Norbury
This son also succeeded Lord Norbury himself on his death in 1831 as 2nd Baron Norbury, whilst his younger brother Daniel succeeded to the viscountcy and
earldom according to the special remainder.

 In 1832, the second Earl also succeeded his elder brother in the two baronies.

 He had already in 1825 assumed the additional surname of Graham by Royal licence.
</item 5>

<item 6>
Earl Cowper
Earl Cowper (pronounced "Cooper") was a title in the Peerage of Great Britain.

 It was created in 1718 by George I for William Cowper, 1st Baron Cowper, his first Lord Chancellor, with remainder in default of male issue of his own to his
younger brother, Spencer Cowper.

 Cowper had already been created Baron Cowper of Wingham in the County of Kent, in the Peerage of England on 14 December 1706, with normal remainder to the
heirs male of his body, and was made Viscount Fordwich, in the County of Kent, at the same time as he was given the earldom, also Peerage of Great Britain and
with similar remainder.

 He was the great-grandson of William Cowper, who was created a Baronet, of Ratling Court in the County of Kent, in the Baronetage of England on 4 March 1642.

 The latter was succeeded by his grandson, the second Baronet.

 He represented Hertford in Parliament.
</item 6>

<item 7>
Copsi
Copsi (or Copsig; Old English: "Cōpsige" ) [died 1067] was a Northumbrian magnate in late Anglo-Saxon England.

 He was a supporter of Tostig, and was exiled along with him in 1065.

 Copsi soon fled to Orkney (then a part of Norway).

 The next year (1066), he joined Tostig at Sandwich, in Kent, with 17 ships.

 Copsi survived Tostig's defeat at Stamford Bridge, and when William the Conqueror prevailed at Hastings he travelled, in March 1067, to pay William homage at
Barking (where William was staying while his tower was being constructed in London).

 In return, William made Copsi earl of Northumbria and sent him back to York.

 Copsi's rule lasted a mere five weeks, at which time he was murdered by Osulf, son of Eadulf III of the ancient Bernician family which had historically
governed the area from Bamburgh, at Newburn-upon-Tyne.
</item 7>

<item 8>
Copsi
Osulf, however, only ruled as earl until that autumn, when he was killed by an outlaw he was tracking.
</item 8>

<item 9>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 9>

<item 10>
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 10>
=== END ITEM id=666093c7677e9e8d ===

=== ITEM id=0e54139faf9f7731 ===
Gold sentences and facts:
Sentence 0: Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.
  Fact 0.0: Cantarella was a poison
  Fact 0.1: Cantarella was allegedly used by the Borgias
  Fact 0.2: The alleged use of Cantarella by the Borgias occurred during the papacy of Pope Alexander VI
Sentence 1:  The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to remove his name from the records.
  Fact 1.0: The use of Cantarella is not well documented in any of the papal records
  Fact 1.1: Cantarella was most likely conceived after 1503
  Fact 1.2: Cantarella was most likely conceived as part of Pope Julius II's effort to remove his name from the records
Sentence 2: Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
  Fact 2.0: Damnatio memoriae is a modern Latin phrase
  Fact 2.1: Damnatio memoriae literally means condemnation of memory
  Fact 2.2: Damnatio memoriae means that a person must not be remembered

Retrieved items:
<item 1>
Sicut Dudum
Sicut Dudum (English: "Just As Long Ago" ) is a papal bull promulgated by Pope Eugene IV in Florence on January 13, 1435, which forbade the enslavement of local
natives in the Canary Islands who had converted or were converting to Christianity.

 "Sicut Dudum" was meant to reinforce "Creator Omnium", issued the previous year, condemning Portuguese slave raids in the Canary Islands.

 Over forty years after "Creator Omnium" and "Sicut Dudum, Pope Sixtus IV found it necessary repeat the prohibition in his papal bull "Regimini Gregis" which
threatened the excommunication of all captains or pirates who enslaved Christians.
</item 1>

<item 2>
Giovanni Borgia (Infans Romanus)
Giovanni Borgia (March 1498 – 1548), known as the Infans Romanus ("the Roman child"), was born into the House of Borgia in secret and is of unclear parentage.

 Speculations of the child's parentage involve either Lucrezia Borgia with her alleged lover, Perotto Calderon or Cesare Borgia, or Pope Alexander VI as his
father.

 Cesare Borgia's biographer Rafael Sabatini says that the truth is fairly clear: Alexander fathered the child with an unknown Roman woman.
</item 2>

<item 3>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 3>

<item 4>
Dum Diversas
Dum Diversas (English: "Until different") is a papal bull issued on 18 June 1452 by Pope Nicholas V.

 It authorized Afonso V of Portugal to conquer Saracens and pagans and consign them to "perpetual servitude".

 Pope Calixtus III reiterated the bull in 1456 with "Inter Caetera" (not to be confused with Alexander VI's), renewed by Pope Sixtus IV in 1481 and Pope Leo X
in 1514 with "Precelse denotionis".

 The concept of the consignment of exclusive spheres of influence to certain nation states was extended to the Americas in 1493 by Pope Alexander VI with "Inter
caetera".
</item 4>

<item 5>
Pope Julius II
Pope Julius II (Italian: "Papa Giulio II" ; Latin: "Iulius II" ) (5 December 1443 – 21 February 1513), born Giuliano della Rovere, and nicknamed "The Fearsome
Pope" and "The Warrior Pope", was Pope from 1 November 1503 to his death in 1513.

 His papacy was marked by an active foreign policy, ambitious building projects, and patronage of the arts—he commissioned the destruction and rebuilding of St.
Peter's Basilica, and Michelangelo's decoration of the ceiling of the Sistine Chapel.

 In addition to an active military policy, he personally led troops into battle on at least two occasions, the first to expel Giovanni Bentivoglio from Bologna
(17 August 1506–23 March 1507), and the second in an attempt to recover Ferrara for the Papal States (1 September 1510–29 June 1512).
</item 5>

<item 6>
Papal conclave, September 1503
The papal conclave of September 1503 elected Pope Pius III to succeed Pope Alexander VI.

 Due to the Italian Wars, the College of Cardinals was surrounded by three potentially hostile armies, loyal to Louis XII of France, Ferdinand II of Aragon, and
Cesare Borgia (the cardinal-nephew of Alexander VI).
</item 6>

<item 7>
The entries cover Greek myth (Despoina, Botres); Renaissance popes and Borgias, including Julius II’s wars, rebuilding of St Peter’s and artistic patronage, and
Felice della Rovere; papal elections, diplomacy and conflicts with emperors; bulls concerning conquest and slavery; major religious texts and councils;
Renaissance paintings by Raphael and Titian; and assorted figures, battles, treaties, and royal successions. They also note disputed histories, including Borgia
poison and
</item 7>

<item 8>
Portrait of Pope Julius II
Portrait of Pope Julius II is an oil painting of 1511–12 by the Italian High Renaissance painter Raphael.

 The portrait of Pope Julius II was unusual for its time and would carry a long influence on papal portraiture.

 From early in its life, it was specially hung at the pillars of the church of Santa Maria del Popolo, on the main route from the north into Rome, on feast and
high holy days.

 Giorgio Vasari, writing long after Julius' death, said that "it was so lifelike and true it frightened everyone who saw it, as if it were the living man
himself".
</item 8>

<item 9>
Dictatus papae
Dictatus papae is a compilation of 27 statements of powers arrogated to the Pope that was included in Pope Gregory VII's register under the year 1075.

 Some historians argue that it was written (or dictated) by Gregory VII himself; others argue that it has been inserted in the register at a later date, and
that it had a different origin.

 In 1087 Cardinal Deusdedit published a collection of decretals, dedicated to Pope Victor III, that embodied the law of the Church – Canon law – which he had
compiled from many sources, both legitimate and false (see Pseudo-Isidore).

 The "Dictatus papae" agrees so clearly and closely with this collection that some have argued the "Dictatus" must have been based on it; and so must be of a
later date of compilation and insertion in the papal register than 1087.
</item 9>

<item 10>
Cantarella
Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.

 It was probably a variation of arsenic or cantharidin powder (made from blister beetles) .

 The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to
remove his name from the records.
</item 10>
=== END ITEM id=0e54139faf9f7731 ===

=== ITEM id=a2454f64456d5085 ===
Gold sentences and facts:
Sentence 0: Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry, famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
  Fact 0.0: Aram Saroyan was born on September 25, 1943.
  Fact 0.1: Aram Saroyan is an American poet.
  Fact 0.2: Aram Saroyan is an American novelist.
  Fact 0.3: Aram Saroyan is an American biographer.
  Fact 0.4: Aram Saroyan is an American memoirist.
  Fact 0.5: Aram Saroyan is an American playwright.
  Fact 0.6: Aram Saroyan is especially known for his minimalist poetry.
  Fact 0.7: A famous example of Aram Saroyan's minimalist poetry is the one-word poem "lighght".
  Fact 0.8: A famous example of Aram Saroyan's minimalist poetry is a one-letter poem comprising a four-legged version of the letter "m".
Sentence 1: Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of microelectronics.
  Fact 1.0: Dr. Edward Keonjian was born on 14 August 1909.
  Fact 1.1: Dr. Edward Keonjian died on 6 September 1999.
  Fact 1.2: Dr. Edward Keonjian was a prominent engineer.
  Fact 1.3: Dr. Edward Keonjian was an early leader in the field of low-power electronics.
  Fact 1.4: Dr. Edward Keonjian was the father of microelectronics.

Retrieved items:
<item 1>
Aram Saroyan
Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry,
famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
</item 1>

<item 2>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 2>

<item 3>
Edward Keonjian
Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of
microelectronics.

 In 1954 Keonjian designed the world's first solar-powered, pocket-sized radio transmitter.

 In 1959 Keonjian designed the first prototype of integrated circuit.

 In 1963 he organized the world's first international symposium on low-power electronics.

 Later on Keonjian collaborated with NASA astronaut Neil Armstrong as chief of failure analysis on the Apollo 11 project.
</item 3>

<item 4>
Levan of Imereti
Levan (Georgian: ლევანი ) (1573–1590), of the Bagrationi Dynasty, was a king of Imereti from 1585 to 1588.

 He succeeded on the death of his father, George II, in 1585 when he was twelve years old.

 With his ascend to the throne, Leon faced a revolt by his own uncle, Constantine, who defied the royal authority and took control of Upper Imereti.

 Leon made an alliance with the Mingrelian prince Mamia IV Dadiani, married his sister Marekhi, and forced Constantine to surrender in 1587.

 A year later, Imereti was invaded by Simon I, the resurgent king of Kartli in eastern Georgia, who sought to reunify all Georgian lands under his crown.

 Leon was forced to flee to the highland province of Lechkhumi, but was soon able to resume the throne after Simon had to return to Kartli.
</item 4>

<item 5>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 5>

<item 6>
Strawberry Saroyan
Strawberry Saroyan (born 1970) is a journalist and author.

 The daughter of Aram Saroyan and granddaughter of playwright William Saroyan and actress Carol Matthau, she spent her childhood in Bolinas, California.

 She has a sister named Cream.

 She writes for the New York Times Style section and is the author of "".
</item 6>

<item 7>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 7>

<item 8>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 8>

<item 9>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 9>

<item 10>
My Name Is Aram
My Name is Aram is a book of short stories by William Saroyan first published in 1940.

 The stories detail the exploits of Aram Garoghlanian, a boy of Armenian descent growing up in Fresno, California, and the various members of his large family.

 This book is assigned reading in some schools.
</item 10>
=== END ITEM id=a2454f64456d5085 ===

=== ITEM id=1b25e6d688a4df1d ===
Gold sentences and facts:
Sentence 0:  After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.
  Fact 0.0: He spent some time in general practice.
  Fact 0.1: He became a Lecturer in animal behaviour at De Montfort University.
  Fact 0.2: He became a Senior Lecturer in animal behaviour at De Montfort University.
  Fact 0.3: He became a Principal Lecturer in animal behaviour at De Montfort University.
  Fact 0.4: He subsequently moved to the University of Lincoln.
  Fact 0.5: He has been Head of the Department of Biological Sciences at the University of Lincoln.
Sentence 1:  Individuals who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title of "Doctor" (often abbreviated "Dr") or, in non-English speaking countries, variants such as "Dr. phil."
  Fact 1.0: People who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title 'Doctor'.
  Fact 1.1: The title 'Doctor' is often abbreviated 'Dr'.
  Fact 1.2: In non-English speaking countries, people with a Doctor of Philosophy degree may use variants such as 'Dr. phil.'

Retrieved items:
<item 1>
Wale Adebanwi
Professor Wale Adebanwi, (born 1969), is a Nigerian - born first Black African Rhodes Professor at Oxford University
</item 1>

<item 2>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 2>

<item 3>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 3>

<item 4>
David M. Sever
David M. Sever is an American herpetologist, histologist, anatomist and reproductive biologist.

 He has been a professor and department head in the Department of Biological Sciences at Southeastern Louisiana University since 2004, and held the Kenneth
Dyson Endowed Professorship in Biological Sciences from 2012 to 2015.

 He is well known for over 30 years of research on the secondary sexual characteristics of salamanders and more generally on comparative histoanatomy of the
urogenital systems of vertebrates.

 and was recognized as the 2013 Distinguished Herpetologist of the Year by the Herpetologists' League.
</item 4>

<item 5>
Wallace L. W. Sargent
Sargent was born in Elsham, North Lincolnshire, the son of a gardener and a housecleaner, and grew up in Winterton, Lincolnshire.

 Sargent was the first person in his family to attend high school, and the first student from his high school to ever attend college.

 He received his bachelor's degree from the University of Manchester in 1956, and his Ph.D. in 1959 from the same institution.
</item 5>

<item 6>
Daniel Mills (biologist)
Daniel Simon Mills, FRCVS (born 21 Aug 1966) is an English biologist and the UK's first Professor of Veterinary Behavioural Medicine based at the University of
Lincoln, United Kingdom.

 He attended St Edmund's College, Ware before studying at the University of Bristol where he received his BVSc degree and completed his Ph.D. degree in animal
behaviour from De Montfort University.

 After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and
subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.

 His research has focused on the assessment of emotion in companion animals and the use of semiochemicals to manage their problem behaviour (pheromonatherapy).
</item 6>

<item 7>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 7>

<item 8>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 8>

<item 9>
Lincoln University College, Malaysia
Lincoln University College, Petaling Jaya, Malaysia, was founded in the year in 2002 as Lincoln College and in 2011 Lincoln College was upgraded to Lincoln
University College.

 Lincoln University College is a private institution that provides higher education, approved by the Malaysian Qualifications Agency (MQA, National
Accreditation Board) and the Ministry of Higher Education.

 Situated in the cosmopolitan town of Petaling Jaya, the university college is close to the capital city, Kuala Lumpur.
</item 9>

<item 10>
William McGinnis
William "Bill" McGinnis, Ph.D. is a molecular biologist and professor of biology at the University of California San Diego.

 At UC San Diego he has also served as the Chairman of the Department of Biology from July 1998 - June 1999, as Associate Dean of the Division of Natural
Sciences from July 1, 1999 - June 2000, and as Interim Dean of the newly established Division of Biological Sciences from July 1, 2000 - February 1, 2001.

 Dr. McGinnis was appointed Dean of the Divisional Biological Sciences on July 1, 2013
</item 10>
=== END ITEM id=1b25e6d688a4df1d ===

=== ITEM id=bbe670aed3179b3b ===
Gold sentences and facts:
Sentence 0: Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.
  Fact 0.0: Like many of Marivaux's plays, La Surprise de l'amour makes use of stock characters from the Commedia dell'arte.
  Fact 0.1: Many of Marivaux's plays make use of stock characters from the Commedia dell'arte.
Sentence 1:  In this play, Arlequin and Columbine are featured.
  Fact 1.0: In La Surprise de l'amour, Arlequin is featured.
  Fact 1.1: In La Surprise de l'amour, Columbine is featured.
Sentence 2: Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the Italian "Commedia dell'arte".
  Fact 2.0: Harlequin is the best-known of the zanni, the comic servant characters from the Italian Commedia dell'arte.
  Fact 2.1: Harlequin is called Arlecchino in Italian.
  Fact 2.2: Harlequin is called Arlequin in French.
  Fact 2.3: Harlequin is called Harlequin in Old French.

Retrieved items:
<item 1>
Pierrot lunaire (book)
Pierrot lunaire: rondels bergamasques ("Moonstruck Pierrot: bergamask rondels") is a collection of fifty poems published in 1884 by the Belgian poet Albert
Giraud (born Emile Albert Kayenburgh), who is usually associated with the Symbolist Movement.

 The protagonist of the cycle is Pierrot, the comic servant of the French Commedia dell'Arte and, later, of Parisian boulevard pantomime.

 The early 19th-century Romantics, Théophile Gautier most notably, had been drawn to the figure by his Chaplinesque pluckiness and pathos, and by the end of the
century, especially in the hands of the Symbolists and Decadents, Pierrot had evolved into an alter-ego of the artist, particularly of the so-called poète
maudit.

 He became the subject of numerous compositions, theatrical, literary, musical, and graphic.
</item 1>

<item 2>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 2>

<item 3>
Gilles (stock character)
Gilles (] )—sometimes Gille—is a stock character of French farce and Commedia dell'Arte.

 He enjoyed his greatest vogue in 18th-century France, in entertainments both at the fairgrounds of the capital and in private and public theaters, though his
origins can be traced back to the 17th century and, possibly, the century previous.

 A "zanni", or comic servant, he is a type of bungling clown, stupid, credulous, and lewd—a character that shares little, problematically, with the sensitive
figure in Watteau's famous portrait that, until the latter half of the 20th century, bore his name alone.

 Gilles fades from view in the 19th century, to persist in the 20th and 21st as the Belgian Gilles of Binche Carnival.
</item 3>

<item 4>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 4>

<item 5>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 5>

<item 6>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 6>

<item 7>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 7>

<item 8>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 8>

<item 9>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 9>

<item 10>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 10>
=== END ITEM id=bbe670aed3179b3b ===

=== ITEM id=ef4efe10460483e4 ===
Gold sentences and facts:
Sentence 0: Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the west.
  Fact 0.0: Longkou was formerly known as Huang County.
  Fact 0.1: Longkou is a port city.
  Fact 0.2: Longkou is in northeastern Shandong province, China.
  Fact 0.3: Longkou faces the Bohai Sea to the north.
  Fact 0.4: Longkou faces the Laizhou Bay to the west.
Sentence 1:  The district faces East China Sea and is connected to Mawei district by the Min River.
  Fact 1.0: The district faces the East China Sea.
  Fact 1.1: The district is connected to Mawei district by the Min River.

Retrieved items:
<item 1>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 1>

<item 2>
Dafeng, Yancheng, is a coastal Jiangsu district with 112 km of shoreline, a salt-making history, protected ecosystems, and China’s largest Père David’s deer
reserve. Changle, Fuzhou, Fujian, has extensive land and sea areas, Tang-era origins, and became a district in 2017. Fuzhoucheng is a Liaoning town near
highways, a reservoir, and a bird reserve. Jiangyin dialect is a Taihu Wu
</item 2>

<item 3>
Xujiacao Changle Station
Xujiacao Changle Station () is a station on Line 1 of the Ningbo Rail Transit that started operations on 30May 2014.

 It is situated over Wangchun Road () and Xinyuan Road () in the Haishu District of Ningbo City, Zhejiang Province, eastern China.
</item 3>

<item 4>
Changle District
(, Foochow Romanized: Diòng-lŏ̤h) is a district located in eastern Fujian province, China.

 Administered by Fuzhou, it occupies a land area of 648 km2 and a sea area of 1327 km2 .

 Changle was established in the sixth year of Emperor Wu-De (623 AD) during the Tang Dynasty, and it became a county-level city on February 18, 1994.

 The district faces East China Sea and is connected to Mawei district by the Min River.

 Due to an increase in businesses, the province is now one of the richest province in China.

 The city was upgraded into a district on August 2017 by a government proposal.
</item 4>

<item 5>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 5>

<item 6>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 6>

<item 7>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 7>

<item 8>
Line 6, Fuzhou Metro
Line 6 of the Fuzhou Metro () or Airport Line () runs from Cangshan District to Changle City.

 It starts at Exhibition Center Station and ends at Fuzhou Changle International Airport Station.

 The total length is 40.9 km.

 Line 6's color is pink.

 The construction is planned to begin by the end of 2016 and ends in 2021, from Exhibition Center to Fuzhou Changle International Airport.
</item 8>

<item 9>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 9>

<item 10>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 10>
=== END ITEM id=ef4efe10460483e4 ===

=== ITEM id=c16a2477d9e506fc ===
Gold sentences and facts:
Sentence 0: MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen neighborhood of Manhattan, New York City.
  Fact 0.0: MiMA stands for 'Middle of Manhattan'.
  Fact 0.1: MiMA is a mixed-use building.
  Fact 0.2: MiMA is located at 450 West 42nd Street.
  Fact 0.3: MiMA is located between Dyer and 10th Avenues.
  Fact 0.4: MiMA is located in the Hell's Kitchen neighborhood of Manhattan, New York City.
Sentence 1:  Ground was broken in 2007 and topping out occurred in early August 2010.
  Fact 1.0: Ground was broken on MiMA in 2007.
  Fact 1.1: Topping out of MiMA occurred in early August 2010.
Sentence 2:  It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on floors 51 to 63, and a Yotel hotel on the lower levels.
  Fact 2.0: MiMA was designed by the Miami-based architecture firm of Arquitectonica.
  Fact 2.1: MiMA has 43 floors of luxury rentals on floors 7 to 50.
  Fact 2.2: MiMA has twelve floors of condominiums on floors 51 to 63.
  Fact 2.3: MiMA has a Yotel hotel on the lower levels.
Sentence 3:  The building's height is 204 meters (670 feet).
  Fact 3.0: The height of MiMA is 204 meters (670 feet).
Sentence 4: 270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan, New York City.
  Fact 4.0: 270 Park Avenue is also known as the JPMorgan Chase Tower.
  Fact 4.1: 270 Park Avenue was formerly known as the Union Carbide Building.
  Fact 4.2: 270 Park Avenue is a high-rise office building.
  Fact 4.3: 270 Park Avenue is located in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
MiMA (building)
MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen
neighborhood of Manhattan, New York City.

 Ground was broken in 2007 and topping out occurred in early August 2010.

 It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on
floors 51 to 63, and a Yotel hotel on the lower levels.

 The building's height is 204 meters (670 feet).
</item 1>

<item 2>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 2>

<item 3>
Robert B. Atwood Building
The Robert B. Atwood Building is an office building located at 550 West 7th Avenue in Downtown Anchorage, Alaska.

 The building houses government offices for the State of Alaska.

 Standing at 20 stories and 81 m (265 ft), it is the second-tallest building in Alaska.

 The building was formerly known as the "Bank of America Center".

 Together with the slightly taller Conoco-Phillips Building, the skyscraper defines Anchorage's skyline.
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
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 5>

<item 6>
Park Avenue House
The Park Avenue House is a high rise residential building located at 2305 Park Avenue in the Park Avenue Historic District in Downtown Detroit, Michigan.

 It was listed on the National Register of Historic Places in 1996.

 It should not be confused with the nearby Park Avenue Hotel, which was demolished in 2015.
</item 6>

<item 7>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 7>

<item 8>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 9>

<item 10>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 10>
=== END ITEM id=c16a2477d9e506fc ===
