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

This file has 10 items.

=== ITEM id=724e637a8f803f9d ===
Gold sentences and facts:
Sentence 0: Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.
  Fact 0.0: Old Smokey is a euphemistic name for the state prison electric chair in New Jersey.
  Fact 0.1: The New Jersey state prison electric chair, called Old Smokey, is on display at the New Jersey State Police Museum.
Sentence 1:  The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
  Fact 1.0: The chair's most notorious victim was Richard Hauptmann.
  Fact 1.1: Richard Hauptmann was executed in the chair.
  Fact 1.2: Richard Hauptmann was found guilty of abducting and killing Charles Augustus Lindbergh, Jr.
  Fact 1.3: The abduction and killing of Charles Augustus Lindbergh, Jr. happened in 1932.
  Fact 1.4: Charles Augustus Lindbergh, Jr. was 20 months old.
  Fact 1.5: Charles Augustus Lindbergh, Jr. was the son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh.
  Fact 1.6: The case is referred to as the Lindbergh kidnapping (see Lindbergh kidnapping).
Sentence 2:  The Lindbergh kidnapping became known as "The Crime of the Century".
  Fact 2.0: The Lindbergh kidnapping became known as 'The Crime of the Century'.

Retrieved items:
<item 1>
The Big House (1930 film)
The Big House is a 1930 American Pre-Code crime drama film directed by George W. Hill, released by Metro-Goldwyn-Mayer, and starring Chester Morris, Wallace
Beery, Lewis Stone and Robert Montgomery.

 The supporting cast features Leila Hyams, George F. Marion, J. C. Nugent, Karl Dane and Tom Kennedy.

 The story and dialogue were written by Frances Marion, with additional dialogue by Joe Farnham and Martin Flavin.

 The story was inspired by a spate of prison riots in 1929 and resulting federal investigation.

 In response, George Hill wrote a twenty-seven page story treatment called "The Reign of Terror: A Story of Crime and Punishment".

 Irving Thalberg gave the go ahead for the screenplay and assigned Frances Marion to work with George Hill.
</item 1>

<item 2>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 2>

<item 3>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 3>

<item 4>
Old Smokey
Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.

 The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus
Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
</item 4>

<item 5>
Cordelia Botkin
Cordelia Botkin (1854 – March 7, 1910) was an American murderer who sent a box of poisoned candy to her ex-lover's wife.

 This was the first American prosecution for a crime which took place in two different jurisdictions, as Botkin had sent the poison from California, but it was
received in Delaware.
</item 5>

<item 6>
Old Sparky
Old Sparky is the nickname of the electric chairs in Arkansas, Connecticut, Florida, Georgia, Illinois, Kentucky, Nebraska, New York, Ohio, Oklahoma, South
Carolina, Texas, Virginia, and West Virginia.

 Old Smokey was the nickname of the electric chairs used in New Jersey, Pennsylvania, and Tennessee.
</item 6>

<item 7>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 7>

<item 8>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 8>

<item 9>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 9>

<item 10>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 10>
=== END ITEM id=724e637a8f803f9d ===

=== ITEM id=53cd261d9e641a99 ===
Gold sentences and facts:
Sentence 0: Mastodon is an American heavy metal band from Atlanta, Georgia.
  Fact 0.0: Mastodon is an American heavy metal band.
  Fact 0.1: Mastodon is from Atlanta, Georgia.
Sentence 1:  Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals in studio (except Bill Kelliher).
  Fact 1.0: Mastodon was formed in 2000.
  Fact 1.1: Mastodon is composed of bassist Troy Sanders.
  Fact 1.2: Mastodon is composed of guitarist Brent Hinds.
  Fact 1.3: Mastodon is composed of guitarist Bill Kelliher.
  Fact 1.4: Mastodon is composed of drummer Brann Dailor.
  Fact 1.5: Troy Sanders, Brent Hinds, and Brann Dailor perform vocals in the studio.
  Fact 1.6: Bill Kelliher does not perform vocals in the studio.
Sentence 2:  Mastodon has released seven studio albums, as well as a number of other records.
  Fact 2.0: Mastodon has released seven studio albums.
  Fact 2.1: Mastodon has released a number of other records.
Sentence 3:  Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.
  Fact 3.0: Mastodon's second full-length release is "Leviathan".
  Fact 3.1: "Leviathan" is a concept album.
  Fact 3.2: "Leviathan" is based on the novel "Moby-Dick".
  Fact 3.3: "Moby-Dick" was written by Herman Melville.
Sentence 4: Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric Erlandson.
  Fact 4.0: Hole was an American alternative rock band.
  Fact 4.1: Hole formed in Los Angeles, California in 1989.
  Fact 4.2: Hole was formed by singer and guitarist Courtney Love.
  Fact 4.3: Hole was formed by lead guitarist Eric Erlandson.

Retrieved items:
<item 1>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 1>

<item 2>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 2>

<item 3>
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
</item 3>

<item 4>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 4>

<item 5>
Meg White
Megan Martha "Meg" White (born December 10, 1974) is an American drummer and occasional singer known for her work with Jack White in the Detroit rock duo The
White Stripes.

 On an impulse, she played on Jack's drums in 1997.

 The two decided to form a band and began performing two months later, calling themselves The White Stripes because of their last name and Meg's affinity for
peppermint candy.

 The band quickly became a Detroit underground favorite, before reaching national, then international fame.

 White has been nominated for various awards as a part of the White Stripes, and has received four Grammy Awards.
</item 5>

<item 6>
Blackdiamondskye
Blackdiamondskye was a 2010 North American concert tour by American bands Alice in Chains, Mastodon, and Deftones.

 Blackdiamondskye is a portmanteau of the three band's most recent albums: Alice in Chains' "Black Gives Way to Blue", Deftones' "Diamond Eyes", and Mastodon's
"Crack the Skye".

 The tour kicked off on September 16 in Chicago and culminated on October 16 in Paradise, Nevada.

 The tour lasted for 19 shows over the course of one month and featured shows at both outdoor amphitheatres and indoor venues.
</item 6>

<item 7>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 7>

<item 8>
Pantera
Pantera was an American heavy metal band from Arlington, Texas.

 The group was formed in 1981 by the Abbott brothers – drummer Vinnie Paul and guitarist Dimebag Darrell – along with lead vocalist Terry Glaze.

 Bassist Rex Brown joined the band the following year, replacing Tommy D. Bradford, who was the unofficial original.

 Having started as a glam metal band, Pantera released four albums during the 1980s.

 Looking for a new and heavier sound, Pantera replaced Glaze with Phil Anselmo in late 1986 and released "Power Metal" in 1988.

 With its fifth album, 1990's "Cowboys from Hell", Pantera introduced a groove metal sound.

 Pantera's sixth album, 1992's "Vulgar Display of Power", exhibited an even heavier sound. "

Far Beyond Driven" (1994) debuted at number one on the "Billboard" 200.
</item 8>

<item 9>
Mastodon (band)
Mastodon is an American heavy metal band from Atlanta, Georgia.

 Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals
in studio (except Bill Kelliher).

 They all perform vocals at live shows.

 Their musical style features progressive concepts and unique instrumentation.

 Mastodon has released seven studio albums, as well as a number of other records.

 The band's debut album, "Remission", released in 2002, garnered significant critical acclaim for its unique sound.

 Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.

 Three magazines awarded the record "Album of the Year" in 2004: "Revolver", "Kerrang!

" and "Terrorizer".
</item 9>

<item 10>
Hole (band)
Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric
Erlandson.

 The band had a revolving line-up of bassists and drummers, their most prolific being drummer Patty Schemel, and bassists Kristen Pfaff (d. 1994) and Melissa
Auf der Maur.
</item 10>
=== END ITEM id=53cd261d9e641a99 ===

=== ITEM id=216ba37e3360aafc ===
Gold sentences and facts:
Sentence 0: Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.
  Fact 0.0: Pennsylvania Route 145 (PA 145) is a state highway.
  Fact 0.1: Pennsylvania Route 145 is 20.89 miles long.
  Fact 0.2: Pennsylvania Route 145 runs north-south.
  Fact 0.3: Pennsylvania Route 145 is in the Lehigh Valley area of eastern Pennsylvania.
Sentence 1:  MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the remainder of the road a four-lane divided highway.
  Fact 1.0: MacArthur Road is a divided highway.
  Fact 1.1: Between U.S. Route 22 (US 22) and Eberhart Road, MacArthur Road is six lanes wide.
  Fact 1.2: Between U.S. Route 22 and Eberhart Road, MacArthur Road has a Jersey barrier.
  Fact 1.3: Between U.S. Route 22 and Eberhart Road, MacArthur Road has jughandles.
  Fact 1.4: The remainder of MacArthur Road is a four-lane divided highway.
Sentence 2:  Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
  Fact 2.0: Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.

Retrieved items:
<item 1>
Pennsylvania Route 145
MacArthur Road is the location of the main commercial center of the Lehigh Valley.

 North of Eagle Point, PA 145 becomes a two-lane undivided road that parallels the Lehigh River, crossing the river into Northampton County at Treichlers.

 The route continues along the east bank of the river and passes through Walnutport before reaching its northern terminus.

 PA 145 is dedicated as the Battle of the Bulge Veterans Memorial Highway in honor of the veterans who fought in the Battle of the Bulge.
</item 1>

<item 2>
Pennsylvania Canal (Delaware Division)
as part of the solution to the United States' first energy crisis.

 The Commonwealth of Pennsylvania built the Delaware canal to feed anthracite stone coal to energy-hungry Philadelphia as part of its transportation
infrastructure building plan known as the Main Line of Public Works—a legislative initiative creating a collection of self-reinforcing internal improvements to
commercial transportation capabilities.
</item 2>

<item 3>
Polyethylene naphthalate
Polyethylene naphthalate (poly(ethylene 2,6-naphthalate or PEN) is a polyester with good barrier properties (even better than Polyethylene terephthalate).

 Because it provides a very good oxygen barrier, it is particularly well-suited for bottling beverages that are susceptible to oxidation, such as beer.

 It is also used in making high performance sailcloth.

 It also has been found to show supreme scintillation properties and is expected to replace classic plastic scintillators.
</item 3>

<item 4>
Jersey barrier
A Jersey barrier or Jersey wall is a modular concrete or plastic barrier employed to separate lanes of traffic.

 It is designed to minimize vehicle damage in cases of incidental contact while still preventing the crossover case of a head-on collision.

 Jersey barriers are also used to reroute traffic and protect pedestrians and workers during highway construction, as well as temporary and semi-permanent
protections against landborne attack such as suicide vehicle bombs.

 A Jersey barrier is also known in the western United States as K-rail, or K-wall, a term borrowed from the California Department of Transportation
specification for temporary concrete traffic barriers, or colloquially as a Jersey bump.

 Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
</item 4>

<item 5>
Breadon Field
Breadon Field was a minor league ballpark in Whitehall Township, Pennsylvania, located on the east side of MacArthur Road, about 0.5 mile north of the U.S.
Route 22 interchange.
</item 5>

<item 6>
Pennsylvania Route 145
Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.

 It connects Interstate 78 (I-78) and PA 309 in Lanark, Lehigh County to PA 248 in Weiders Crossing, Northampton County.

 PA 145 is the main north-south arterial into Allentown, the third-largest city in the state.

 The route enters the city on South 4th Street and follows multiple streets to downtown, where it follows the one-way pair of 6th Street northbound and 7th
Street southbound.

 North of Allentown in Whitehall Township, a seven-mile (11.2 km) portion of PA 145 is known as MacArthur Road, named in honor of General Douglas MacArthur.

 MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the
remainder of the road a four-lane divided highway.

 MacArthur Road is the location of the main commercial center of the Lehigh Valley.
</item 6>

<item 7>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 7>

<item 8>
Hanbys Corner, Delaware
Hanbys Corner is an unincorporated community in New Castle County, Delaware, United States.

 It is located at the intersection of Delaware Route 3 (Marsh Road) and Delaware Route 92 (Naamans Road), in Brandywine Hundred.

 The area is named for Richard G. Hanby, who first purchased the 125 acre parcel from the descendants of William Penn in 1753.

 His descendants included several notable figures in the local political scene including Samuel Winfield Hanby (1817-1892) who was elected as a State
Representative in 1874, Jacob Klose Hanby (1839-1932) who was Samuel's son and was elected State Representative in 1904, Robert Johnson Hanby (1834-1898), who
served in the 124th PA infantry during the Civil War and was elected State Senator in 1896, and Florence Wood Hanby (1870-1963), who was Robert's
daughter-in-law and was the first woman elected to the Delaware House of Representatives in 1924.

 In addition Wayne Hanby and James Hanby have both served as Justices of the Peace for New Castle County.
</item 8>

<item 9>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 9>

<item 10>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 10>
=== END ITEM id=216ba37e3360aafc ===

=== ITEM id=33e87df937794070 ===
Gold sentences and facts:
Sentence 0:  His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".
  Fact 0.0: The exploits of the man referred to here in that battle were described by Hal Moore.
  Fact 0.1: Hal Moore described the exploits in the book 'We Were Soldiers Once and Young'.
Sentence 1: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 1.0: Harold Gregory 'Hal' Moore, Jr. was born on February 13, 1922.
  Fact 1.1: Harold Gregory 'Hal' Moore, Jr. died on February 10, 2017.
  Fact 1.2: Harold Gregory 'Hal' Moore, Jr. was a United States Army lieutenant general.
  Fact 1.3: Harold Gregory 'Hal' Moore, Jr. was an author.

Retrieved items:
<item 1>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 1>

<item 2>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 2>

<item 3>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 3>

<item 4>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 4>

<item 5>
The text profiles military leaders across the U.S. and Allied services. James Winnefeld was U.S. vice chairman of the Joint Chiefs (2011–15); Matthew Ridgway
led airborne forces in WWII; Arthur Tedder and naval officers Arthur Power and Algernon Willis helped plan or execute Allied Mediterranean campaigns. Hal Moore
and Myron Diduryk served at Ia Drang. George Finch built the Air National Guard; Mark Mitchell earned the Distinguished Service Cross in Afghanistan. Other
</item 5>

<item 6>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 6>

<item 7>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 7>

<item 8>
Myron F. Diduryk
Myron F. Diduryk (July 15, 1938 to April 24, 1970) was a Ukrainian-American United States Army Major, who played a key role as an infantry company commander in
the Battle of Ia Drang, the first major battle of the Vietnam War.

 His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".

 Moore said that Diduryk was, “… the finest battlefield company commander I had ever seen, bar none.”

 Diduryk was killed in action on his second tour in Vietnam.
</item 8>

<item 9>
A military-reference compilation profiles U.S. and Allied commanders, special forces, intelligence officials, organizations, ranks, honors, and memorials. It
surveys major battles and campaigns from the Civil War and World War II—including the Pacific, Mediterranean, Normandy, and Germany—as well as Ia Drang and
later operations. Topics also include WWII casualties, the EA-18G Growler and Australian airbases, naval ships and carriers, and weapons ranging from the
Vickers gun to Pershing II
</item 9>

<item 10>
William K. MacNulty
William Kirk MacNulty (May 22, 1892 - August 3, 1964) was a U. S. Marine with a long and distinguished record.

 He was a young U.S. Marine Corps Second Lieutenant during World War One and saw action at the Battle of the Argonne Forest.

 He served as a captain during the Second U.S. Nicaraguan Campaign (1926–33).

 During the Second World War as Lieutenant Colonel he commanded the U.S. Marine Corps defense of Guam against overwhelming Imperial Japanese forces during the
First Battle of Guam.

 He was incarcerated by the Japanese as a prisoners of war.

 He was promoted to Brigadier General during captivity and retired from military service in 1946.

 He is buried at the Golden Gate National Cemetery in San Bruno, San Mateo County, California.
</item 10>
=== END ITEM id=33e87df937794070 ===

=== ITEM id=7e7cb9200b61634a ===
Gold sentences and facts:
Sentence 0:  She was also a member of a secret society composed exclusively of Cheyenne women.
  Fact 0.0: The woman in question was also a member of a secret society.
  Fact 0.1: The secret society was composed exclusively of Cheyenne women.
Sentence 1: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.
  Fact 1.0: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance.
  Fact 1.1: The other central institution of traditional Cheyenne Indian tribal governance is the Council of Forty-four.

Retrieved items:
<item 1>
Concho Indian Boarding School
Concho Indian Boarding School (also known as the Cheyenne-Arapaho Boarding School at Concho or Concho Indian School and home to the Concho Demonstration School)
was a boarding school for members of the Cheyenne and Arapaho Tribes and later opened to other Native American students.

 It existed from 1909 to 1983.

 It was located in central Oklahoma, approximately 1 mile south of Concho, Oklahoma and 4 miles north of El Reno, Oklahoma.

 The name of the town and school is the Spanish word for "shell" and was named for the Indian agent, Charles E. Shell.
</item 1>

<item 2>
Graph query row:
person: Ehyophsta
institution: Council of Forty-four
</item 2>

<item 3>
Council of Forty-four
The Council of Forty-four was one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the military societies such
as the Dog Soldiers.

 The influence of the Council of Forty-four waned in the face of internal conflict among the Cheyenne about Cheyenne policy toward encroaching white settlers on
the Great Plains, and was dealt a severe blow by the Sand Creek Massacre.
</item 3>

<item 4>
Cheyenne
The Cheyenne ( ) are one of the Indigenous peoples of the Great Plains and their language is of the Algonquian language family.

 The Cheyenne comprise two Native American tribes, the Só'taeo'o or Só'taétaneo'o (more commonly spelled as Suhtai or Sutaio) and the Tsétsêhéstâhese (also
spelled Tsitsistas).

 These tribes merged in the early 19th century.

 Today, the Cheyenne people are split into two federally recognized Nations: the Southern Cheyenne, who are enrolled in the Cheyenne and Arapaho Tribes in
Oklahoma, and the Northern Cheyenne, who are enrolled in the Northern Cheyenne Tribe of the Northern Cheyenne Indian Reservation in Montana.
</item 4>

<item 5>
Cheyenne military societies
Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.

 While council chiefs are responsible for overall governance of individual bands and the tribe as a whole, the headmen of military societies are in charge of
maintaining discipline within the tribe, overseeing tribal hunts and ceremonies, and providing military leadership.

 Historically, council chiefs selected which of the six military societies would assume these duties; after a period of time on-duty, the chiefs would select a
different society to take up the duties.
</item 5>

<item 6>
Chief Dull Knife College
Chief Dull Knife College (originally Dull Knife Memorial College) is a small, open-admission, Native American tribal community college and land grant
institution.

 Located on the Northern Cheyenne Indian Reservation, in the city of Lame Deer, it has a current enrollment of about 141 students.

 On average, more than half of its graduates move on to four-year colleges.
</item 6>

<item 7>
Eugene Little Coyote
Eugene Little Coyote was the president of the Northern Cheyenne Indian Reservation from 2004 to 2007.

 He was elected in November 2004, defeating the incumbent president, Geri Small.

 However, after a conflict with reservation vice president Rick Wolfname that began in July 2007 escalated, the Northern Cheyenne tribal council declared that
it would oust Little Coyote as the tribe's president.

 Little Coyote was forcibly removed from office and arrested for trespassing in the tribal office building on December 28, 2007.

 It is worth noting that the move by the tribal council to remove Little Coyote from office was deemed unconstitutional and void by the Northern Cheyenne
Constitutional Court.

 The Constitutional Court was however overruled by Bureau of Indian Affairs' regional director Ed Parisian, and later the Assistant Secretary of Indian Affairs
Carl J. Artman.

 Their involvement in this matter set a historical precedence of the BIA directly meddling with tribal governments.
</item 7>

<item 8>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 8>

<item 9>
Ehyophsta
Ehyophsta (Cheyenne for "Yellow-Haired Woman") was a Cheyenne woman.

 She was the daughter of Stands-in-the-Timber who died in 1849, and she was the niece of Bad Faced Bull.

 She fought in the Battle of Beecher Island in 1868, and also fought the Shoshone that same year, where she counted coup against one enemy and killed another.

 She fought the Shoshone again in 1869.

 She was also a member of a secret society composed exclusively of Cheyenne women.

 She died in 1915.
</item 9>

<item 10>
Northern Cheyenne Indian Reservation
The Northern Cheyenne Indian Reservation (Tsėhéstáno in Cheyenne, formerly named the Tongue River Indian Reservation) is home of the federally recognized
Northern Cheyenne Tribe.

 Located in southeastern Montana, the reservation is approximately 444,000 acres in size and home to approximately 5,000 Cheyenne people.

 The tribal and government headquarters are in Lame Deer, which is also the home of the annual Northern Cheyenne Pow wow.

 The reservation is bounded on the east by the Tongue River and on the west by the Crow Reservation.

 There are small parcels of non-contiguous off-reservation trust lands in Meade County, South Dakota, northeast of the city of Sturgis.

 Its timbered ridges that extend into northwestern South Dakota are part of Custer National Forest and it is approximately 40 mi east of the site of the 1876
Battle of the Greasy Grass (known to most Americans as the Battle of the Little Bighorn).
</item 10>
=== END ITEM id=7e7cb9200b61634a ===

=== ITEM id=341e5bd338da21a0 ===
Gold sentences and facts:
Sentence 0: Sir Ridley Scott (born 30 November 1937) is an English film director and producer.
  Fact 0.0: Sir Ridley Scott was born on 30 November 1937.
  Fact 0.1: Sir Ridley Scott is English.
  Fact 0.2: Sir Ridley Scott is a film director.
  Fact 0.3: Sir Ridley Scott is a film producer.
Sentence 1:  Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
  Fact 1.0: Ridley Scott had a commercial breakthrough with the science-fiction horror film Alien, released in 1979.
  Fact 1.1: Ridley Scott's best known works include the neo-noir dystopian science fiction film Blade Runner, released in 1982.
  Fact 1.2: Ridley Scott's best known works include the historical drama Gladiator, released in 2000.
  Fact 1.3: Gladiator won the Best Picture Oscar.
  Fact 1.4: Ridley Scott's best known works include the science fiction film The Martian, released in 2015.
Sentence 2:  A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work behind the camera, with Griffith and Joseph Henabery as his mentors.
  Fact 2.0: Ridley Scott was a collaborator of D.W. Griffith.
  Fact 2.1: Ridley Scott appeared in The Birth of a Nation, released in 1915.
  Fact 2.2: Ridley Scott appeared in Intolerance, released in 1916.
  Fact 2.3: Ridley Scott gave up acting in 1917.
  Fact 2.4: Ridley Scott gave up acting to concentrate on work behind the camera.
  Fact 2.5: Griffith was a mentor of Ridley Scott.
  Fact 2.6: Joseph Henabery was a mentor of Ridley Scott.

Retrieved items:
<item 1>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 1>

<item 2>
Enrico Cocozza
Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.

 His often surreal films were mainly filmed in and around the town of Wishaw in Scotland, where his family owned the popular Belhaven Cafe.

 These include "Chick's Day" (1950), a prize winner at the 1951 Scottish Amateur Film Festival, "The Living Ghost" (1957), and "Glasgow's Docklands" (1959).

 Illness later forced him to give up making films and he spent most of his working life teaching at the University of Strathclyde in Glasgow.

 In 2001 he was the subject of the Channel 4 documentary "Artery: the Story of Enrico Cocozza".
</item 2>

<item 3>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 3>

<item 4>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 4>

<item 5>
Graph query row:
person: Ridley Scott
connected_works: ['The Polar Bears', 'Alien', 'Blade Runner', 'Gladiator', 'The Martian']
output_count: 5
compared_person: Elmer Clifton
compared_works: ['The Flame of Youth', 'Seven Doors to Death', 'Gangsters of the Frontier']
compared_output_count: 3
comparison: Ridley Scott has more
</item 5>

<item 6>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 6>

<item 7>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 7>

<item 8>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 8>

<item 9>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 9>

<item 10>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 10>
=== END ITEM id=341e5bd338da21a0 ===

=== ITEM id=7ff4f5f063268408 ===
Gold sentences and facts:
Sentence 0: of Montreal is an American experimental pop band from Athens, Georgia.
  Fact 0.0: of Montreal is an American band.
  Fact 0.1: of Montreal is an experimental pop band.
  Fact 0.2: of Montreal is from Athens, Georgia.
Sentence 1: Vib Gyor was a band that was formed in Leeds, England.
  Fact 1.0: Vib Gyor was a band.
  Fact 1.1: Vib Gyor was formed in Leeds, England.

Retrieved items:
<item 1>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 1>

<item 2>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 2>

<item 3>
Graph query row:
band: Vib Gyor
formation_location_relationship: LOCATED_IN
formation_place: Leeds
location_relationship: LOCATED_IN
location: England
</item 3>

<item 4>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 4>

<item 5>
The Gorgeous
The Gorgeous were a Canadian metalcore band formed in the summer of 2003 in Montreal.

 The band formed with Dali Shaw on guitar, Julien Brousseau on Bass and Miguel Shaw on drums.

 The three played a few shows without a vocalist and quickly recruited Jordon Daniel.
</item 5>

<item 6>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 6>

<item 7>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 7>

<item 8>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 8>

<item 9>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 9>

<item 10>
Fallen (Vib Gyor song)
"Fallen" was the first single from the Leeds band Vib Gyor.

 It was first given limited release in July 2006, and has attained critical acclaim and a large amount of radio play on both sides of the atlantic.
</item 10>
=== END ITEM id=7ff4f5f063268408 ===

=== ITEM id=5772a8a389cc4b2c ===
Gold sentences and facts:
Sentence 0:  Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.
  Fact 0.0: Players compete in races.
  Fact 0.1: The race tracks are inspired by attractions such as the Haunted Mansion.
  Fact 0.2: The race tracks are inspired by attractions such as Pirates of the Caribbean.
  Fact 0.3: The race tracks are inspired by attractions such as Space Mountain.
  Fact 0.4: The race tracks are inspired by attractions such as Big Thunder Mountain Railroad.
  Fact 0.5: Players race on tracks to acquire missing parts for the park's fireworks machine.
  Fact 0.6: The park's fireworks machine was accidentally destroyed by Chip 'n' Dale.
  Fact 0.7: Chip 'n' Dale destroyed the fireworks machine while they were gathering acorns.
Sentence 1:  Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.
  Fact 1.0: Space Mountain opened on January 15, 1975.
  Fact 1.1: Space Mountain is the oldest operating roller coaster in the state of Florida.
  Fact 1.2: Space Mountain is the original version of an iconic attraction.
  Fact 1.3: The original version of Space Mountain has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide.
  Fact 1.4: The replication of Space Mountain excludes the Shanghai Disneyland Resort.

Retrieved items:
<item 1>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 1>

<item 2>
Space Mountain
Space Mountain is the name of a space-themed indoor roller coaster attraction located at five of the Magic Kingdom-style Disney Parks (all except for Shanghai
Disneyland Park).

 Although all five versions of the attraction are different in nature, all have a similar domed exterior façade that is a landmark for the respective park.

 The first Space Mountain ride opened in 1975 at Walt Disney World Resort in Florida, and similar attractions were then eventually built at the other Disney
parks.
</item 2>

<item 3>
Space Mountain (Disneyland)
Space Mountain (known as Hyperspace Mountain from 2015 to 2017) is an indoor, space-themed roller coaster in Tomorrowland at Disneyland in Anaheim, California.

 Opened on May 27, 1977, it was the second roller coaster built at Disneyland, and was the second of the five versions of Space Mountain built by The Walt
Disney Company.

 Its exterior façade is one of Disneyland's four "mountain" structures that serve as park landmarks.
</item 3>

<item 4>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 4>

<item 5>
Walt Disney World Dolphin
The Walt Disney World Dolphin is a resort hotel designed by architect Michael Graves located between Epcot and Disney's Hollywood Studios in the Walt Disney
World Resort in Bay Lake, Florida, next to Disney's BoardWalk Resort area.

 It opened on June 1, 1990 and is joined to its sister hotel, the Walt Disney World Swan (also designed by Graves) by a palm-tree lined covered walkway crossing
a lagoon.

 The Walt Disney World Swan and Dolphin is a joint venture between the Walt Disney Company, Tishman Hotel Corporation, MetLife and Starwood Hotels and Resorts.

 The land the resort occupies is owned by the Walt Disney Company, while the buildings themselves are leased by Disney to the Tishman Hotel Corporation and
MetLife but operated by Starwood Hotels & Resorts Worldwide under the Sheraton Hotels brand.

 The Walt Disney World Swan and Dolphin are a part of the Walt Disney Collection of resorts; because of this they are Disney branded and guests of the resort
have access to special Disney benefits available to Disney Resort Hotel guests only.
</item 5>

<item 6>
Space Mountain (Magic Kingdom)
Space Mountain is an indoor dark outer space-themed steel roller coaster at the Magic Kingdom theme park at the Walt Disney World Resort in Lake Buena Vista,
Florida.

 Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction
that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.

 RCA helped fund Space Mountain's construction and sponsored the ride from 1975 to 1993; FedEx sponsored Space Mountain from 1994 to 2004.
</item 6>

<item 7>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 7>

<item 8>
Best Western Lake Buena Vista Resort Hotel
The Best Western Lake Buena Vista Resort Hotel is a 325-room resort that is on the property of Walt Disney World Resort in Lake Buena Vista, Florida.

 The resort is located across from the Disney Springs area.

 The hotel originally opened in November 1972 as the Travelodge at Lake Buena Vista.

 It was operated by Travelodge from 1972 through 1983, then in 1984 the resort became the Viscount Hotel and operated as such until 1988.

 In 1989, it was reacquired by Travelodge.

 In 2000, the Travelodge hotel switched over to become the Best Western Lake Buena Vista Resort Hotel.

 The hotel was renovated in 2004 and has 2 outdoor pools.
</item 8>

<item 9>
Rock 'n' Roller Coaster Starring Aerosmith
Rock 'n' Roller Coaster Starring Aerosmith is an enclosed launched steel roller coaster at Disney's Hollywood Studios at the Walt Disney World Resort and at the
Walt Disney Studios Park in Disneyland Paris (where it is called "Rock 'n' Roller Coaster avec Aerosmith").

 As the attraction's name suggests, the coaster features Aerosmith members, Steven Tyler, Joe Perry, Tom Hamilton, Joey Kramer, and Brad Whitford.
</item 9>

<item 10>
Walt Disney World Quest: Magical Racing Tour
Walt Disney World Quest: Magical Racing Tour is a 2000 go-kart racing video game based on attractions at the Walt Disney World Resort.

 Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain
Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.

 The game was developed by Crystal Dynamics and published by Eidos Interactive.

 The Game Boy Color version was developed by Prolific Publishing.
</item 10>
=== END ITEM id=5772a8a389cc4b2c ===

=== ITEM id=bd65a64782f82e69 ===
Gold sentences and facts:
Sentence 0: The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in 1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
  Fact 0.0: The Karlsschrein, English name Shrine of Charlemagne, is in Aachen Cathedral.
  Fact 0.1: The Karlsschrein was made in Aachen.
  Fact 0.2: The Karlsschrein was made at the command of Frederick II, Holy Roman Emperor.
  Fact 0.3: The Karlsschrein was completed in 1215.
  Fact 0.4: Frederick II's grandfather was Frederick Barbarossa.
  Fact 0.5: Frederick Barbarossa exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen.
  Fact 0.6: Frederick Barbarossa exhumed Charlemagne's bones in 1165.
  Fact 0.7: The exhumation of Charlemagne's bones happened before the Karlsschrein was completed.
Sentence 1: Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the Holy Roman Emperor from 1155 until his death.
  Fact 1.0: Frederick I's German name is "Friedrich I".
  Fact 1.1: Frederick I's Italian name is "Federico I".
  Fact 1.2: Frederick I lived from 1122 to 10 June 1190.
  Fact 1.3: Frederick I is also known as Frederick Barbarossa.
  Fact 1.4: Frederick Barbarossa's Italian name is "Federico Barbarossa".
  Fact 1.5: Frederick I was the Holy Roman Emperor from 1155 until his death.

Retrieved items:
<item 1>
Battle of Cingoli
The Battle of Cingoli was fought in 1250 between the forces of the Holy Roman Empire and the armies of the Guelphs and the Papal States, the area being so
notable due to its nickname as "The Balcony of Marche".

 The Imperial forces inflicted a crushing defeat on the Pope's factions.

 After the battle, Frederick II, Holy Roman Emperor was smitten with leprosy, and Pope Innocent IV declared the illness to be an act of God.

 This was most probably due to the Emperor's excommunication; a common threat and practice for subduing overconfident Catholic leadership.
</item 1>

<item 2>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 2>

<item 3>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 3>

<item 4>
Papal election, 1241
The papal election, following the death of Pope Gregory IX, from September 21 to October 25, 1241 elected Cardinal Goffredo da Castiglione as Pope Celestine IV.

 The election took place during the first of many protracted "sede vacantes" of the Middle Ages, and like many of them was characterized by disputes between
popes and the Holy Roman Emperor.

 Specifically, the election took place during the war between Frederick II, Holy Roman Emperor against the Lombard League and the deceased pontiff, Pope Gregory
IX, with Italy divided between pro-Papal and pro-Imperial factions known as the Guelphs and Ghibellines.
</item 4>

<item 5>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 5>

<item 6>
List of Holy Roman Empresses
Holy Roman Empress or Empress of the Holy Roman Empire is the title given to the consort (wife) or regent of the Holy Roman Emperor.

 The elective dignity of Holy Roman Emperor was restricted to males only, therefore there was never a Holy Roman Empress regnant, though women such as Theophanu
or Maria Theresa of Austria, who controlled the power of rule, served as de facto Empresses regnant.
</item 6>

<item 7>
Battle of Parma
The Battle of Parma was fought on 18 February 1248 between the forces of Holy Roman Emperor Frederick II and the Guelphs.

 The Guelphs attacked the Imperial camp when Frederick II was away.

 The Imperial forces were defeated and much of Frederick's treasure was lost.
</item 7>

<item 8>
Karlsschrein
The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in
1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
</item 8>

<item 9>
Constance, Queen of Sicily
Constance (2 November 1154 – 27 November 1198) was Queen regnant of Sicily in 1194–98, jointly with her spouse from 1194 to 1197, and with her infant son
Frederick II, Holy Roman Emperor, in 1198, as the heiress of the Norman kings of Sicily.

 She was also Holy Roman Empress by marriage to Henry VI, Holy Roman Emperor.
</item 9>

<item 10>
Frederick I, Holy Roman Emperor
Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the
Holy Roman Emperor from 1155 until his death.

 He was elected King of Germany at Frankfurt on 4 March 1152 and crowned in Aachen on 9 March 1152.

 He became King of Italy in 1155 and was crowned Roman Emperor by Pope Adrian IV on 18 June 1155.

 Two years later, the term "sacrum " ("holy") first appeared in a document in connection with his Empire.

 He was later formally crowned King of Burgundy, at Arles on 30 June 1178.

 He was named "Barbarossa " by the northern Italian cities which he attempted to rule: Barbarossa means "red beard" in Italian; in German, he was known as
"Kaiser Rotbart ", which has the same meaning.
</item 10>
=== END ITEM id=bd65a64782f82e69 ===

=== ITEM id=93f43bbdb67128c9 ===
Gold sentences and facts:
Sentence 0: "Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the majority of their members quit and formed Faith No More.
  Fact 0.0: "Quiet in Heaven" / "Song of Liberty" is a double A-side single.
  Fact 0.1: "Quiet in Heaven" / "Song of Liberty" is the only official release by Faith No Man.
  Fact 0.2: Faith No Man is stylised as Faith•No Man•.
  Fact 0.3: The single was released before the majority of Faith No Man's members quit.
  Fact 0.4: After the majority of Faith No Man's members quit, they formed Faith No More.
Sentence 1:  Before settling on their current name in 1982, the band performed under the names Sharp Young Men and later Faith No Man.
  Fact 1.0: The band settled on its current name in 1982.
  Fact 1.1: Before settling on its current name in 1982, the band performed under the name Sharp Young Men.
  Fact 1.2: The band later performed under the name Faith No Man.

Retrieved items:
<item 1>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 1>

<item 2>
Quiet in Heaven/Song of Liberty
"Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the
majority of their members quit and formed Faith No More.

 The songs have not appeared in any of the band's compilation albums and have not been performed live since Mike Morris left the band.
</item 2>

<item 3>
Jefferson's Tree of Liberty
Jefferson's Tree of Liberty is the tenth album by Jefferson Starship, released on September 2, 2008.

 It is the band's first studio album since 1999's "Windows of Heaven".

 The new album includes cover songs from Irish, American, English, and Latin-American traditions.

 The title is a reference to Thomas Jefferson's quotation, "The tree of liberty must be refreshed from time to time with the blood of patriots and tyrants."

 The idea began in 2003 as "The Cuba Project", which was to include classic protest and folk songs recorded in Cuba.

 In 2008 the album was finally recorded but in California.

 About half of the songs planned for The Cuba Project were used on the final cut, with other songs coming from Jefferson Starship's previous repertoire and
another project band member Paul Kantner had planned called "On the Threshold of Fire."

 The promotional tour for the album began in late June with shows at Larkspur, California followed by tours in the US and Europe before the album's release, and
continued through December 2008 with a further tour in the US and a tour in Japan.
</item 3>

<item 4>
Million Dollar Quartet
"Million Dollar Quartet" is a recording of an impromptu jam session involving Elvis Presley, Jerry Lee Lewis, Carl Perkins, and Johnny Cash made on December 4,
1956, at the Sun Record Studios in Memphis, Tennessee.

 An article about the session was published in the "Memphis Press-Scimitar" under the title "Million Dollar Quartet".

 The recording was first released in Europe in 1981 as "The Million Dollar Quartet" with 17 tracks.

 A few years later more tracks were discovered and released as "The Complete Million Dollar Session".

 In 1990, the recordings were released in the United States as "Elvis Presley - The Million Dollar Quartet".

 This session is considered a seminal moment in rock and roll.
</item 4>

<item 5>
Corcovado (song)
"Corcovado" (known in English as "Quiet Nights of Quiet Stars") is a bossa nova song written by Antônio Carlos Jobim in 1960.

 An English lyric was later written by Gene Lees.

 The Portuguese title refers to the Corcovado mountain in Rio de Janeiro.

 Andy Williams recorded the song with English lyrics, reaching #92 in the "Billboard" Hot 100 and #18 in the Hot Adult Contemporary Tracks chart in 1965.

 Also receiving air-play, contemporaneously with Andy Williams' recording of "Quiet Nights," was Kitty Kallen's version.

 Her album, titled "Quiet Nights," was released by 20th Century-Fox Records in 1964.
</item 5>

<item 6>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 6>

<item 7>
The Promised Land (Sagol 59 album)
A live show recording was released digitally in April 2017 as "The Promised Land at The Deanery, Brooklyn NY 3-23-2017".

 Live in Cafe Bialik, Tel Aviv, a soundboard concert recording, was released July 4th, 2017.
</item 7>

<item 8>
No One Talks
"No One Talks" was the first commercial single taken from Grammy award-winning singer, Moya Brennan's album "Signature" released the same year.

 This was Moya's first single available to download from more than one online shop.

 The B-side to the single, the traditional Gaelic song "Éirigh Suas a Stóirín (Rise Up My Love)" was previously only available on the Germany-only album "Óró -
A Live Session".

 The cover shows a photograph by Mella Travers.
</item 8>

<item 9>
Refugees of the Heart
Refugees of the Heart is the sixth solo studio album by Steve Winwood, released in 1990.

 The album contained the hit single, "One and Only Man", which topped the Mainstream Rock Tracks chart and saw the return of former Traffic bandmate Jim Capaldi
to Winwood's songwriting team.

 By coincidence, a Traffic reunion followed in 1994.

 Because of that collaboration, Winwood would not record another solo album until late 1997.

 Also "I Will Be Here" and "Another Deal Goes Down" were released as singles.

 Winwood stated about the closing track, “In The Light Of Day”: 'When Will and I wrote the song (.

.)

 it was our idea of what Nelson Mandela's dream was, while he was in prison.

 It was really just a fantasy of ours, but that’s what we based the song on.'
</item 9>

<item 10>
List of songs recorded by Faith No More
Faith No More, a San Francisco-based band, have recorded approximately 100 songs over the course of their career; this includes material from six studio albums,
one live album, and numerous B-side tracks and out-takes.

 Faith No More were founded in 1981 as Sharp Young Men, and changed their name to Faith No Man before releasing the 1982 double A-side single "Quiet in Heaven"
/ "Song of Liberty".

 Upon assuming the name Faith No More, the band's first two full-length albums, "We Care a Lot" and "Introduce Yourself", were driven mostly by new vocalist
Chuck Mosley and the "metallic guitar" of Jim Martin, blending elements of rap and heavy metal music.

 Mosley was later replaced by Mr. Bungle vocalist Mike Patton, who added lyrics to the already-written music for 1989's "The Real Thing".

 Their next studio album, "Angel Dust", moved away from the band's rap-influenced sound to experiment with different musical genres; a trend which became much
more pronounced on the 1995 album "King for a Day... Fool for a Lifetime".
</item 10>
=== END ITEM id=93f43bbdb67128c9 ===
