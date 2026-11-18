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

=== ITEM id=929d6b860e87cb0f ===
Gold sentences and facts:
Sentence 0: Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.
  Fact 0.0: Marc Predka was born April 21, 1975.
  Fact 0.1: Marc Predka is better known by his stage name Tha Trademarc.
  Fact 0.2: Marc Predka is an American hip hop artist.
Sentence 1:  He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance theme "The Time Is Now".
  Fact 1.0: Marc Predka first came to prominence when he collaborated with his first cousin, John Cena.
  Fact 1.1: John Cena is a WWE wrestler.
  Fact 1.2: Marc Predka and John Cena collaborated on the 2005 album 'You Can't See Me'.
  Fact 1.3: The album 'You Can't See Me' includes Cena's entrance theme 'The Time Is Now'.
Sentence 2: "The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.
  Fact 2.0: 'The Time Is Now' is a song by John Cena.
  Fact 2.1: John Cena is a professional wrestler.
  Fact 2.2: John Cena is an actor.
  Fact 2.3: John Cena is a hip hop musician.

Retrieved items:
<item 1>
Motion City Soundtrack, a Minneapolis rock band formed in 1997, released six studio albums over nearly 20 years, blending pop punk/emo with prominent Moog
synthesizers. Justin Pierre wrote most of the band’s 107 recorded songs, including nine covers. Key releases include *Commit This to Memory* (2005), produced by
Mark Hoppus; *Even If It Kills Me* (2007), shaped by Pierre’s struggles with addiction; *My Dinosaur Life* (
</item 1>

<item 2>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 2>

<item 3>
Rebuild (song)
"Rebuild" is a song written by Jon Foreman of Switchfoot and Matt Thiessen of Relient K.

 The song was written in conjunction with the bands' 2007 Appetite for Construction Tour, featuring members from all three bands on tour, Switchfoot, Relient K,
and Ruth.

 Thiessen sang second vocals, and Dustin Ruth of Ruth played harmonica.
</item 3>

<item 4>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 4>

<item 5>
Relient K’s “My Girlfriend,” written by 15-year-old Matt Thiessen, began as “Marilyn Manson Ate My Girlfriend”; it reflects a friend’s rejection of
Christianity. Thiessen also co-wrote Switchfoot’s “Rebuild” with Jon Foreman. Fuel’s 2007 *Angels & Devils* was Toryn Green’s only album with the band; 2014’s
*Puppet Strings* returned Brett Scallions but lacked original members Carl Bell and
</item 5>

<item 6>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 6>

<item 7>
Tha Trademarc
Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.

 He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance
theme "The Time Is Now".

 He also appeared in the music videos of the songs, "Bad Bad Man" and "Right Now" with Cena.

 Prior to the album, Trademarc appeared on Cena's first WWE DVD "Word Life".

 He now works as a classroom specialist for the Peace Corps of America.
</item 7>

<item 8>
Borat’s “In My Country There Is Problem” is an antisemitic comedy song featured in a U.S. bar segment. Toše Proeski’s 2004 double album *Den Za Nas* collected
Eurovision selections, including “Life” (14th in the final), plus new songs. “Can We Talk” has numerous covers, samples, TV uses, and compilation appearances.
Marc Ribot’s 2005 *Spiritual Unity* honors Albert Ayler. Sig
</item 8>

<item 9>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 9>

<item 10>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 10>
=== END ITEM id=929d6b860e87cb0f ===

=== ITEM id=ef591a79239e4ae2 ===
Gold sentences and facts:
Sentence 0: "I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.
  Fact 0.0: "I'll Keep the Lovelight Burning" is a popular song.
  Fact 0.1: "I'll Keep the Lovelight Burning" was written in 1942.
  Fact 0.2: "I'll Keep the Lovelight Burning" was written by Harry Tobias.
  Fact 0.3: "I'll Keep the Lovelight Burning" was written by Nick Kenny.
  Fact 0.4: "I'll Keep the Lovelight Burning" was written by Harold Levey.
  Fact 0.5: "I'll Keep the Lovelight Burning" was popularized by Patti Page.
  Fact 0.6: Patti Page popularized "I'll Keep the Lovelight Burning" in 1949.
Sentence 1: Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country music.
  Fact 1.0: Clara Ann Fowler's date range is November 8, 1927 to January 1, 2013.
  Fact 1.1: Clara Ann Fowler is known by the professional name Patti Page.
  Fact 1.2: Patti Page was an American singer.
  Fact 1.3: Patti Page was a singer of traditional pop music.
  Fact 1.4: Patti Page was a singer of country music.

Retrieved items:
<item 1>
You Keep Me Hangin' On
"You Keep Me Hangin' On" is a 1966 song written and composed by Holland–Dozier–Holland.

 It first became a popular "Billboard" Hot 100 number one hit for the American Motown group The Supremes in late 1966.

 The rock band Vanilla Fudge covered the song a year later and had a top ten hit with their version.

 British pop singer Kim Wilde covered "You Keep Me Hangin' On" in 1986, bumping it back to number one on the "Billboard" Hot 100 in June 1987.

 The single reached number one by two different musical acts in America.

 In the first 32 years of the "Billboard" Hot 100 rock era, “You Keep Me Hangin' On” became one of only six songs to achieve this feat.

 In 1996, country music singer Reba McEntire's version reached number 2 on the US "Billboard" Hot Dance Club Play chart.
</item 1>

<item 2>
I'll Keep the Lovelight Burning
"I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.

 Louis Armstrong also covered the song in 1949.
</item 2>

<item 3>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 3>

<item 4>
Patti Page
Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country
music.

 She was the top-charting female vocalist and best-selling female artist of the 1950s, selling over 100 million records during a six-decade long career.

 She was often introduced as "the Singin' Rage, Miss Patti Page".

 New York WNEW disc-jockey William B. Williams introduced her as "A Page in my life called Patti".
</item 4>

<item 5>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 5>

<item 6>
Keep This Fire Burning
"Keep This Fire Burning" is a song by Swedish pop singer Robyn, released as the first single from her third album "Don't Stop the Music".

 It was released in Sweden on September 21, 2002, where it became her highest charting single since 1995's "Do You Really Want Me (Show Respect)".

 The song was also released as a single in Australia under the name "By Your Side", due to the Australian bush fires which were happening at the time.

 In 2008, a re-recorded version of the song appeared on the special edition of Robyn's eponymous album.
</item 6>

<item 7>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 7>

<item 8>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 8>

<item 9>
Breathing Your Love
"Breathing Your Love" is a song by the Swedish singer Darin featuring vocals by singer Kat DeLuna and the first single from "Flashback".

 The song co-written by Darin with RedOne, Bilal Hajji and Novel was released to radio stations and as a digital download in Sweden on 8 October 2008.

 The single was released in Finland and is also the first single from Darin to be released in the United Kingdom in January 2010.

 In October 2009 Swedish astronaut Christer Fuglesang took 2 copies of the single into space with him, as his daughter is a Darin fan, a photo of the disc can
be seen with earth in the distance can be found on the official Darin website.
</item 9>

<item 10>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 10>
=== END ITEM id=ef591a79239e4ae2 ===

=== ITEM id=1438e0df1ab7cdeb ===
Gold sentences and facts:
Sentence 0: Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.
  Fact 0.0: Eenasul Fateh was born on 3 April 1959.
  Fact 0.1: Eenasul Fateh's Bengali name is ঈনাসুল ফাতেহ.
  Fact 0.2: Eenasul Fateh is also known by his stage name Aladin.
  Fact 0.3: Eenasul Fateh is a Bangladeshi-British cultural practitioner.
  Fact 0.4: Eenasul Fateh is a magician.
  Fact 0.5: Eenasul Fateh is a live artist.
  Fact 0.6: Eenasul Fateh is a former international management consultant.
Sentence 1: Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans for improvement.
  Fact 1.0: Management consulting is the practice of helping organizations improve their performance.
  Fact 1.1: Management consulting operates primarily through the analysis of existing organizational problems.
  Fact 1.2: Management consulting operates primarily through the development of plans for improvement.

Retrieved items:
<item 1>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 1>

<item 2>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 2>

<item 3>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 3>

<item 4>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 4>

<item 5>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 5>

<item 6>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 6>

<item 7>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 7>

<item 8>
Lawrence A. Gordon
Lawrence A. Gordon is the EY Alumni Professor of Managerial Accounting and Information Assurance at the University of Maryland’s Robert H. Smith School of
Business.

 He is also an Affiliate Professor in the University of Maryland Institute for Advanced Computer Studies.

 Dr. Gordon earned his Ph.D. in Managerial Economics from Rensselaer Polytechnic Institute.

 An internationally known scholar in the areas of managerial accounting (often called management accounting) and cybersecurity economics, Dr. Gordon's research
focuses on such issues as economic aspects of information security (including cybersecurity or computer security), corporate performance measures, cost
management systems, and capital investments.

 He is the author of approximately 100 articles, published in such journals as The Accounting Review, ACM Transactions on Information and System Security,
Communications of the ACM, Journal of Financial and Quantitative Analysis, Journal of Computer Security, MIS Quarterly, Accounting, Organizations and Society,
and the Journal of Accounting and Public Policy.

 Dr. Gordon's current research emphasizes the importance of applying concepts from economics and managerial accounting to an information-based economy.

 Dr.
</item 8>

<item 9>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 9>

<item 10>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 10>
=== END ITEM id=1438e0df1ab7cdeb ===

=== ITEM id=beda549d4e1aea08 ===
Gold sentences and facts:
Sentence 0:  The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss' 1983 book "Fatal Vision".
  Fact 0.0: The journalist in question is Joe McGinniss.
  Fact 0.1: The murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald.
  Fact 0.2: Dr. Jeffrey R. MacDonald became the subject of Joe McGinniss' 1983 book 'Fatal Vision'.
Sentence 1: Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in February 1970.
  Fact 1.0: Jeffrey Robert MacDonald was born on October 12, 1943.
  Fact 1.1: Jeffrey Robert MacDonald is an American medical doctor.
  Fact 1.2: Jeffrey Robert MacDonald was convicted in 1979 of murdering his pregnant wife.
  Fact 1.3: Jeffrey Robert MacDonald was convicted in 1979 of murdering his two daughters.
  Fact 1.4: The murders took place in February 1970.

Retrieved items:
<item 1>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 1>

<item 2>
The entries cover pornographic publisher Auguste Brancart; Albert Camus, his absurdist philosophy, Nobel Prize, fiction and essays opposing nihilism and capital
punishment; and his biographers. Other topics include the disputed Smiley Face murder theory; Joe McGinniss, Jeffrey MacDonald and the Fatal Vision controversy;
the Guantánamo memoir Enemy Combatant; New Zealand’s Parker–Hulme case and Mazengarb Report; the Lindbergh kidnapping; Nazi collaborator Christian de
</item 2>

<item 3>
Jeffrey R. MacDonald
Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in
February 1970.
</item 3>

<item 4>
Dibdin and Bickerstaffe’s 1768 opera *The Padlock* succeeded through Mungo, a blackface role later altered in revival. German fan series *Starship Highlander*
parodied sci-fi from 1995. *The Ipcress File* (1965) earned acclaim and a BAFTA. “Fatal Vision” refers to intoxication-simulation goggles, a 1984 miniseries,
McGinniss’s controversial 1983 true-crime book
</item 4>

<item 5>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 5>

<item 6>
Fatal Vision (miniseries)
Fatal Vision is a 1984 American television miniseries based on the account, in the book of the same name, of the murders in 1970 at Fort Bragg of the wife and
daughters of U.S. Army officer Jeffrey R. MacDonald.
</item 6>

<item 7>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 7>

<item 8>
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 8>

<item 9>
Fatal Vision (disambiguation)
Fatal Vision is the 1983 true crime book by Joe McGinniss which lies at the center of the "Fatal Vision" controversy.
</item 9>

<item 10>
Fatal Vision controversy
The controversy over Fatal Vision, journalist and author Joe McGinniss's best-selling 1983 true crime book, is a decades-long dispute spanning several court
cases and discussed in several other published works.
</item 10>
=== END ITEM id=beda549d4e1aea08 ===

=== ITEM id=d445b8885e10d5bd ===
Gold sentences and facts:
Sentence 0:  Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.
  Fact 0.0: Most figures are shaped with the body of Mickey Mouse.
  Fact 0.1: Most figures have different themed markings.
  Fact 0.2: Most figures have different themed colors.
  Fact 0.3: Most figures have different themed patterns.
Sentence 1:  He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.
  Fact 1.0: Mickey Mouse was created by Walt Disney.
  Fact 1.1: Mickey Mouse was created by Ub Iwerks.
  Fact 1.2: Mickey Mouse was created at the Walt Disney Studios.
  Fact 1.3: Mickey Mouse was created in 1928.

Retrieved items:
<item 1>
Donald Duck
Donald Duck is a cartoon character created in 1934 at Walt Disney Productions.

 Donald is an anthropomorphic white duck with a yellow-orange bill, legs, and feet.

 He typically wears a sailor shirt and cap with a bow tie.

 Donald is most famous for his semi-intelligible speech and his mischievous and temperamental personality.

 Along with his friend Mickey Mouse, Donald is one of the most popular Disney characters and was included in TV Guide's list of the 50 greatest cartoon
characters of all time in 2002.

 He has appeared in more films than any other Disney character, and is the most published comic book character in the world outside of the superhero genre.
</item 1>

<item 2>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 2>

<item 3>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 3>

<item 4>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 4>

<item 5>
Felix the Cat
Felix the Cat is a funny-animal cartoon character created in the silent film era.

 The anthropomorphic black cat with his black body, white eyes, and giant grin, coupled with the surrealism of the situations in which his cartoons place him,
combine to make Felix one of the most recognized cartoon characters in film history.

 Felix was the first character from animation to attain a level of popularity sufficient to draw movie audiences.
</item 5>

<item 6>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 6>

<item 7>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 7>

<item 8>
Vinylmation
Vinylmation is a brand of 1.5", 3", and 9" vinyl collectibles sold at Disney theme parks, select Disney Stores, and the online Disney Store.

 The name "Vinylmation" is a combination of the word "Animation" and "Vinyl".

 Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.

 This means that all figures have round mouse ears, whether or not the painted character has them.

 This is accommodated by painting the ears with backgrounds - stars, stripes, rainbows - to match the painted character.

 The series exception is the Park Starz series which does not resemble Mickey at all, and more closely resembles iconic figures from the parks.

 Vinylmation was first introduced in July 2008, and the first figures were introduced in November 2008.
</item 8>

<item 9>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 9>

<item 10>
Mickey Mouse
Mickey Mouse is a funny animal cartoon character and the official mascot of The Walt Disney Company.

 He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.

 An anthropomorphic mouse who typically wears red shorts, large yellow shoes, and white gloves, Mickey has become one of the world's most recognizable
characters.
</item 10>
=== END ITEM id=d445b8885e10d5bd ===

=== ITEM id=fe059320c7d89fc2 ===
Gold sentences and facts:
Sentence 0: David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
  Fact 0.0: David Ross Lederman was born December 12, 1894.
  Fact 0.1: David Ross Lederman died August 24, 1972.
  Fact 0.2: David Ross Lederman was an American film director.
  Fact 0.3: David Ross Lederman was noted for his Western, action, and adventure films of the 1930s and 1940s.
Sentence 1: Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
  Fact 1.0: Stefan Ruzowitzky was born 25 December 1961.
  Fact 1.1: Stefan Ruzowitzky is an Austrian film director.
  Fact 1.2: Stefan Ruzowitzky is an Austrian screenwriter.

Retrieved items:
<item 1>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 1>

<item 2>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 2>

<item 3>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 3>

<item 4>
Austrian director-screenwriter Stefan Ruzowitzky made *The Inheritors* (1998), Austria’s unsuccessful Oscar submission, and *The Counterfeiters* (2007), a WWII
drama about Nazi forgery at Sachsenhausen that won the foreign-language-film Oscar; Devid Striesow played Bernhard Krüger. German director Wolfgang Becker is
known for *Good Bye, Lenin!* (2003), starring Daniel Brühl and Katrin Saß. Other entries
</item 4>

<item 5>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 5>

<item 6>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 6>

<item 7>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 7>

<item 8>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 8>

<item 9>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 9>

<item 10>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 10>
=== END ITEM id=fe059320c7d89fc2 ===

=== ITEM id=b559526caf58ab40 ===
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
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
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
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 3>

<item 4>
Nashville Tribute Band
The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
</item 4>

<item 5>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 5>

<item 6>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 6>

<item 7>
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
</item 7>

<item 8>
The Hillmen (album)
The Hillmen is a studio album by The Hillmen, a southern California bluegrass band originally known as The Golden State Boys.

 The Hillmen consisted of Chris Hillman (later of The Byrds, The Flying Burrito Brothers, Manassas and The Desert Rose Band) on mandolin, country
singer/songwriter Vern Gosdin on guitar and lead vocals, his brother Rex Gosdin on double bass, and Don Parmley (later of the Bluegrass Cardinals) on banjo.
</item 8>

<item 9>
Grizzly River Run
Grizzly River Run is located at Disney California Adventure at the Disneyland Resort in Anaheim, California.

 It is similar to Kali River Rapids in Disney's Animal Kingdom but distinctive as the rafts are engineered to spin as they descend chutes.

 The attraction's name comes from Grizzly Peak, the bear shaped mountain that the rapids flow around.

 It was designed by Walt Disney Imagineering and constructed by Intamin.
</item 9>

<item 10>
The Pilgrims (band)
The Pilgrims are an alternative rock band from Windsor, Vermont.

 Seven Days contributor Dan Bolles compared their style to alt-punk and indie rock, while John Powell, writing for www.angelica-music.com, has likened them to
rock and roll bands from the 1960s and 1970s.

 They are part of the Windsor, VT music collective What Doth Life, which has released albums by The Pilgrims, Giant Travel Avant Garde, Derek and The Demons,
Carton, and Luke Chrisinger.
</item 10>
=== END ITEM id=b559526caf58ab40 ===

=== ITEM id=746e6051e5493353 ===
Gold sentences and facts:
Sentence 0: The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
  Fact 0.0: The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran.
  Fact 0.1: The collision occurred on Monday, February 8, 1993.
  Fact 0.2: The collision killed all 133 people on board both aircraft involved.
  Fact 0.3: One aircraft involved was an Iran Air Tours Tupolev Tu-154M, registered EP-ITD.
  Fact 0.4: The other aircraft involved was an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
Sentence 1: Iran Air Tours is an airline based in Tehran, Iran.
  Fact 1.0: Iran Air Tours is an airline.
  Fact 1.1: Iran Air Tours is based in Tehran, Iran.

Retrieved items:
<item 1>
1993 Auckland mid-air collision
The 1993 Auckland mid-air collision was an aircraft accident in New Zealand.

 It occurred on 26 November 1993, when two aircraft operated by Airwork, under contract to the New Zealand Police, collided and crashed in central Auckland.

 The mid-air collision of the Aérospatiale TwinStar helicopter and Piper Archer aeroplane resulted in the deaths of all four occupants – a civilian Airwork
pilot on each aircraft and two New Zealand Police officers on the helicopter.

 The accident occurred in daylight with excellent visibility, in uncontrolled airspace (class G), with both aircraft flying under visual flight rules.

 Both the helicopter and aeroplane were operated by Airwork (NZ), and working under contract to the New Zealand Police at the time of the accident.
</item 1>

<item 2>
Iran Air Tours
Iran Air Tours is an airline based in Tehran, Iran.

 The airline mainly operates under the English name Iran Airtour (i.e. without the plural "s") as written in Persian (Persian: ایران ایر تور‎ ‎ ).

 This error is due to confusion between different pluralization rules in English and Persian.

 Iran Air Tours was established as a subsidiary of Iran Air and operates scheduled domestic services and international services in the Middle East as well as
Eastern Europe, as well as charter services.
</item 2>

<item 3>
1948 Northwood mid-air collision
The 1948 Northwood mid-air collision took place on 4 July at 15:03 when a Douglas DC-6 of Scandinavian Airlines System (SAS) and an Avro York C.1 of the Royal
Air Force (RAF) collided in mid-air over Northwood in London, UK.

 All thirty-nine people aboard both aircraft were killed.

 It was SAS's first fatal aviation accident and was at the time the deadliest civilian aviation accident in the UK.

 It is still the deadliest mid-air collision in British history.
</item 3>

<item 4>
1955 Cincinnati mid-air collision
The 1955 Cincinnati mid-air collision occurred on January 12, 1955, when a Trans World Airlines Martin 2-0-2 on takeoff from Boone County Airport (now the
Cincinnati/Northern Kentucky International Airport) collided in mid-air with a privately owned Douglas DC-3 that had entered the airport's control space without
proper clearance.

 There were no survivors.
</item 4>

<item 5>
1963 Aeroflot Tupolev Tu-124 Neva river ditching
The 1963 Aeroflot Tupolev Tu-124 Neva river ditching was a water landing by a Tupolev Tu-124 of the Soviet state airline "Aeroflot" (Moscow division).

 The aircraft took off from Tallinn-Ülemiste Airport (TLL) at 08:55 on 21 August with 45 passengers and 7 crew on board.

 The aircraft (registration number SSSR-45021) was built in 1962 and was scheduled to fly to Moscow–Vnukovo (VKO) under the command of 27-year-old captain
Victor Mostovoy.

 After takeoff the nose gear did not retract.

 Ground control diverted the flight to Leningrad (LED) – because of fog at Tallinn.
</item 5>

<item 6>
1993 Tehran mid-air collision
The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both
aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
</item 6>

<item 7>
Charkhi Dadri mid-air collision
The Charkhi Dadri mid-air collision occurred on 12 November 1996 over the village of Charkhi Dadri, to the west of New Delhi, India.

 The aircraft involved were a Saudi Arabian Airlines Boeing 747-100B en route from Delhi to Dhahran, Saudi Arabia, and a Kazakhstan Airlines Ilyushin Il-76 en
route from Chimkent, Kazakhstan, to Delhi.

 The crash killed all 349 people on board both planes, making it the world's deadliest mid-air collision, the deadliest aviation accident to occur in India, and
the third-deadliest aircraft accident in the history of aviation, behind only the Tenerife airport disaster and Japan Airlines Flight 123.
</item 7>

<item 8>
Bukovyna Airlines
Bukovyna Airlines, also known as Bukovyna Aviation Enterprise, is a charter airline based in Chernivtsi, Ukraine, operating chartered passenger flights out of
Chernivtsi International Airport.

 The company was founded in 1999.

 In 2013 Bukovyna was one of two Ukrainian airlines that had sanctions imposed against them by the US government.

 Bukovyna Airlines was hit with sanctions because it was leasing its US-built McDonnell Douglas MD-80 series aircraft to Iranian airlines Mahan Air and Iran
Air.

 The Iranian airlines were themselves under sanction by the US government.
</item 8>

<item 9>
Traffic collision avoidance system
A traffic collision avoidance system or traffic alert and collision avoidance system (both abbreviated as TCAS, and pronounced tee-kas) is an aircraft collision
avoidance system designed to reduce the incidence of mid-air collisions between aircraft.

 It monitors the airspace around an aircraft for other aircraft equipped with a corresponding active transponder, independent of air traffic control, and warns
pilots of the presence of other transponder-equipped aircraft which may present a threat of mid-air collision (MAC).

 It is a type of airborne collision avoidance system mandated by the International Civil Aviation Organization to be fitted to all aircraft with a maximum
take-off mass (MTOM) of over 5700 kg or authorized to carry more than 19 passengers.

 CFR 14, Ch I, part 135 requires that TCAS I is installed for aircraft with 10-30 passengers and TCAS II for aircraft with more than 30 passengers.
</item 9>

<item 10>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 10>
=== END ITEM id=746e6051e5493353 ===

=== ITEM id=a0fa3231063e2307 ===
Gold sentences and facts:
Sentence 0: Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television host/personality who "TV Guide" called "The Charlie Chaplin of Television".
  Fact 0.0: Alan Young was born Angus Young.
  Fact 0.1: Alan Young was born on November 19, 1919.
  Fact 0.2: Alan Young died on May 19, 2016.
  Fact 0.3: Alan Young was a British-born Canadian-American actor.
  Fact 0.4: Alan Young was a voice actor.
  Fact 0.5: Alan Young was a comedian.
  Fact 0.6: Alan Young was a radio and television host/personality.
  Fact 0.7: TV Guide called Alan Young "The Charlie Chaplin of Television".
Sentence 1:  Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983) and in various other films, TV series and video games until his death.
  Fact 1.0: Alan Young was the voice of Disney's Scrooge McDuck for over thirty years.
  Fact 1.1: Alan Young first voiced Scrooge McDuck in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983).
  Fact 1.2: Alan Young voiced Scrooge McDuck in various other films, TV series and video games.
  Fact 1.3: Alan Young voiced Scrooge McDuck until his death.
Sentence 2:  During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy Awards in 1951.
  Fact 2.0: During the 1940s and 1950s, Alan Young starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television.
  Fact 2.1: The television version of "The Alan Young Show" gained Alan Young two Emmy Awards in 1951.
Sentence 3:  The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.
  Fact 3.0: The cartoon is an adaptation of Charles Dickens' "A Christmas Carol".
  Fact 3.1: The cartoon stars Scrooge McDuck as Ebenezer Scrooge.

Retrieved items:
<item 1>
A Christmas Carol (2009 film)
A Christmas Carol is a 2009 American 3D computer animated motion-capture fantasy film written and directed by Robert Zemeckis.

 It is an adaptation of the Charles Dickens story of the same name and stars Jim Carrey in a multitude of roles, including Ebenezer Scrooge as a young,
middle-aged, and old man, and the three ghosts who haunt Scrooge.

 The film also features supporting roles done by Gary Oldman, Colin Firth, Bob Hoskins, Robin Wright, and Cary Elwes.
</item 1>

<item 2>
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
</item 2>

<item 3>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 3>

<item 4>
Scrooge, or, Marley's Ghost
Scrooge, or, Marley's Ghost is a 1901 British short silent drama film, directed by Walter R. Booth, featuring the miserly Ebenezer Scrooge confronted by
Marley's ghost and given visions of Christmas past, present, and future, is the earliest known film adaptation of Charles Dickens's 1843 novel "A Christmas
Carol".

 The film, "although somewhat flat and stage-bound to modern eyes," according to Ewan Davidson of BFI Screenonline, "was an ambitious undertaking at the time,"
as, "not only did it attempt to tell an 80 page story in five minutes, but it featured impressive trick effects, superimposing Marley's face over the door
knocker and the scenes from his youth over a black curtain in Scrooge's bedroom."
</item 4>

<item 5>
Scrooge (1913 film)
Scrooge is a 1913 British black and white silent film based on the 1843 novel "A Christmas Carol" by Charles Dickens.

 It starred Seymour Hicks as Ebenezer Scrooge.

 In the United States it was released in 1926 as "Old Scrooge".
</item 5>

<item 6>
Scar (Disney)
Scar is a fictional character who appears in Walt Disney Pictures' 32nd animated feature film "The Lion King" (1994).

 The character is voiced by English actor Jeremy Irons, while his singing voice is provided by both Irons and American actor Jim Cummings, the latter of whom
was hired to replace Irons when the former damaged his singing voice.

 Subsequently, Scar makes minor appearances in the film's sequel " "(1998) and "The Lion King 1½ "(2004), in both of which he is voiced entirely by Cummings, as
well as appearing in the Broadway musical adaptation of the film, in which the role of Scar was originated by American actor John Vickery.
</item 6>

<item 7>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 7>

<item 8>
Pinocchio (1957 TV-musical)
Pinocchio, a 1957 television production of "Pinocchio", is a live (and live-action) musical version starring Mickey Rooney in the title role of the puppet who
wishes to become a real boy.

 Based on the novel by Carlo Collodi which also inspired the classic Walt Disney animated film, this version featured a now-forgotten new score by Alec Wilder
and William Engvick.

 It was telecast once on NBC as a television special, and, as far as is known, never rebroadcast by NBC, or even restaged with a different cast as was Rodgers
and Hammerstein's "Cinderella".

 Nor has it ever been issued on VHS or DVD.

 Other notable actors who appeared in the special included Walter Slezak (as Geppetto), Fran Allison (as the Blue Fairy), Martyn Green (as the Fox), Jerry
Colonna (as a Ringmaster), and Stubby Kaye as a Town Crier, a role he repeated in Wilder and Engvick's 1958 television musical, "Hansel and Gretel".

 "Pinocchio" was directed by noted Broadway choreographer Hanya Holm.
</item 8>

<item 9>
Scrooge (1935 film)
Scrooge is a 1935 British fantasy film directed by Henry Edwards and starring Seymour Hicks, Donald Calthrop and Robert Cochran.

 Hicks appears as Ebenezer Scrooge, the miser who hates Christmas.

 It was the first sound version of the Charles Dickens classic "A Christmas Carol", not counting a 1928 short subject that now appears to be lost.

 Hicks had previously played the role of Scrooge on the stage many times beginning in 1901, and again in a 1913 British silent film version.
</item 9>

<item 10>
Scrooge (1951 film)
Scrooge is a 1951 British fantasy drama film and an adaptation of Charles Dickens' "A Christmas Carol" (1843).

 It stars Alastair Sim as Ebenezer Scrooge, and was produced and directed by Brian Desmond Hurst, with a screenplay by Noel Langley.

 It was released as A Christmas Carol in the United States.
</item 10>
=== END ITEM id=a0fa3231063e2307 ===

=== ITEM id=1551b8b526227a1c ===
Gold sentences and facts:
Sentence 0: Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.
  Fact 0.0: Claudio Giovanni Antonio Monteverdi was baptized on 15 May 1567.
  Fact 0.1: Claudio Giovanni Antonio Monteverdi died on 29 November 1643.
  Fact 0.2: Claudio Giovanni Antonio Monteverdi was an Italian composer.
  Fact 0.3: Claudio Giovanni Antonio Monteverdi was an Italian string player.
  Fact 0.4: Claudio Giovanni Antonio Monteverdi was an Italian choirmaster.
Sentence 1:  A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
  Fact 1.0: Monteverdi composed both secular and sacred music.
  Fact 1.1: Monteverdi was a pioneer in the development of opera.
  Fact 1.2: Monteverdi is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
Sentence 2: Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.
  Fact 2.0: Georges Bizet was born on 25 October 1838.
  Fact 2.1: Georges Bizet died on 3 June 1875.
  Fact 2.2: Georges Bizet was registered at birth as Alexandre César Léopold Bizet.
  Fact 2.3: Georges Bizet was a French composer.
  Fact 2.4: Georges Bizet was a composer of the romantic era.
Sentence 3:  Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the most popular and frequently performed works in the entire opera repertoire.
  Fact 3.0: Georges Bizet is best known for his operas.
  Fact 3.1: Bizet's career was cut short by his early death.
  Fact 3.2: Bizet achieved few successes before his final work.
  Fact 3.3: Bizet's final work is 'Carmen'.
  Fact 3.4: Carmen has become one of the most popular works in the entire opera repertoire.
  Fact 3.5: Carmen is one of the most frequently performed works in the entire opera repertoire.

Retrieved items:
<item 1>
Roma Symphony (Bizet)
The Symphony in C "Roma" is the second of Georges Bizet's symphonies.

 Unlike his first symphony, also in C major, which was written quickly at the age of 17, "Roma" was written over an eleven-year span, between the ages of 22 and
33 (he died at age 36).

 Bizet was never fully satisfied with it, subjecting it to a number of revisions, but died before finishing his definitive version.

 All four movements were performed in his lifetime, but never all on the same occasion.

 The full symphony in its latest revision was premiered in 1875, after his death.

 It is perhaps because of Bizet's dissatisfaction that the work is often said to be "unfinished".

 However, in the form in which it exists today, it is a complete work and is fully scored.

 It has been recorded a number of times but is not often heard on the concert platform.
</item 1>

<item 2>
The material surveys Italian opera history and related figures. Verdi’s Busseto theatre opened in 1868, seats 300, and occupies the former Pallavicino fortress.
Verdi studied under Ferdinando Provesi from age 11; French composer Ambroise Thomas wrote *Mignon* and *Hamlet*, whose 1868 premiere featured bass Armand
Castelmary, later famed for dying onstage in New York. Monteverdi, a Renaissance-to-Baroque pioneer,
</item 2>

<item 3>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 3>

<item 4>
Georges Bizet
Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.

 Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the
most popular and frequently performed works in the entire opera repertoire.
</item 4>

<item 5>
Symphony in C (Bizet)
Soon thereafter, Bizet's first British biographer Douglas Charles Parker (1885–1970) showed the manuscript to the conductor Felix Weingartner, who led the first
performance in Basel, Switzerland, on 26 February 1935.
</item 5>

<item 6>
Symphony in C (Bizet)
The Symphony in C is an early work by the French composer Georges Bizet.

 According to "Grove's Dictionary", the symphony "reveals an extraordinarily accomplished talent for a 17-year-old student, in melodic invention, thematic
handling and orchestration."

 Bizet started work on the symphony in C major on 29 October 1855, four days after turning 17, and finished it roughly a month later.

 It was written while he was studying at the Paris Conservatoire under the composer Charles Gounod, and was evidently a student assignment.

 Bizet showed no apparent interest in having it performed or published, and the piece was never played in his lifetime.

 He used certain material from the symphony in later works, however.

 There is no mention of the work in Bizet's letters, and it was unknown to his earlier biographers.

 His widow, Geneviève Halévy (1849–1926), gave the manuscript to Reynaldo Hahn, who left it along with other papers to the archives of the conservatory library,
where it was found in 1933 by Jean Chantavoine.
</item 6>

<item 7>
The entries span opera, music, theatre, literature and Renaissance history. Bernhard Bötel sang roles from Mozart to Verdi; operas range from Gluck and Hasse to
Rossini and later composers. Commedia dell’arte’s masked stock characters, improvisation and lazzi are described. Musical highlights include Tchaikovsky’s
Second Piano Concerto, Debussy’s *Masques* and Bartók’s quartets. Other topics include Verdi’s Busseto theatre,
</item 7>

<item 8>
John Whenham
John Whenham is an English musicologist and academic who specializes in early Italian baroque music.

 He earned both a Bachelor of Music and a Master of Music from the University of Nottingham, and a Doctor of Philosophy from the University of Oxford.

 He is a leading expert on the life and works of Claudio Monteverdi, and is the author of the books "Duet and Dialogue in the Age of Monteverdi" (Ann Arbor,
Michigan: University Microfilms International, 1982) "Monteverdi, 'Orfeo' " (London: Cambridge University Press, 1986), "Monteverdi, Vespers (1610)" (Cambridge
University Press, 1997), and "The Cambridge Companion to Monteverdi" (with Richard Wistreich, Cambridge University Press, 2007).

 For five years he was co-editor of the journal "Music & Letters".

 He currently serves on the board of the Birmingham Early Music Festival and is head of the music history department at the University of Birmingham.
</item 8>

<item 9>
Bernhard Bötel’s roles ranged from Mozart’s Belmonte to Verdi’s Duke of Mantua. The entries survey operas across eras: Gluck’s reform works *Orfeo ed Euridice*
(1762) and *Paride ed Elena* (1770); Hasse’s *Romolo ed Ersilia* (1765); Rossini’s *Mosè in Egitto* (1818); and later works by Donizetti, Bernstein,
</item 9>

<item 10>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 10>
=== END ITEM id=1551b8b526227a1c ===

=== ITEM id=63300f3900f6b229 ===
Gold sentences and facts:
Sentence 0:  The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
  Fact 0.0: The women's events include the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park.
  Fact 0.1: The women's events include the Roxy Pro at Sunset Beach.
  Fact 0.2: The women's events include the Billabong Pro Maui at Honolua Bay, Maui.
Sentence 1: Haleʻ iwa is a North Shore community and census-designated place (CDP) in the Waialua District of the island of Oʻ ahu, City and County of Honolulu.
  Fact 1.0: Haleʻiwa is a North Shore community.
  Fact 1.1: Haleʻiwa is a census-designated place (CDP).
  Fact 1.2: Haleʻiwa is in the Waialua District of the island of Oʻahu.
  Fact 1.3: Haleʻiwa is in the City and County of Honolulu.

Retrieved items:
<item 1>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 1>

<item 2>
World Match Racing Tour
The World Match Racing Tour is an annual match racing tour and World Championship in the match racing event of sailing.

 The World Match Racing Tour has "special event"-status from the International Sailing Federation.

 The Tour awards over US$1.5 million in prize money, with points awarded at each event culminating in the crowning of the ISAF Match Racing World Champion.

 A US$500,000 overall prize pool for the Championship is now on offer.
</item 2>

<item 3>
2012 Monterrey Open
The 2012 Monterrey Open was a women's tennis tournament played on outdoor hard courts.

 It was the 4th edition of the Monterrey Open and was an International tournament on the 2012 WTA Tour.

 It took place at the Sierra Madre Tennis Club in Monterrey, Mexico, from 20 to 26 February.

 Tímea Babos won the singles title.
</item 3>

<item 4>
Ocean City Open
The Ocean City Open is a defunct tennis tournament that was played on the Grand Prix tennis circuit in 1977.

 The event was held in Ocean City, Maryland.

 Vitas Gerulaitis won the singles title while Alex Metreveli and Bill Scanlon partnered to win the doubles title.
</item 4>

<item 5>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 5>

<item 6>
Andy Irons
Philip Andrew Irons (July 24, 1978November 2, 2010) was an American professional surfer.

 Irons learned to surf on the dangerous and shallow reefs of the North Shore in Kauai, Hawaii.

 Over the course of his professional career, he won three world titles (2002, 2003, 2004), three Quiksilver Pro France titles (2003, 2004, 2005), two Rip Curl
Pro Search titles (2006 and 2007) and 20 elite tour victories including the Vans Triple Crown of Surfing four times from 2002-2006.

 On September 3, 2010 he won the Billabong Pro Teahupoo in Tahiti.

 He and his family hosted the Annual Irons Brothers Pinetrees Classic, a contest for youngsters.

 The Governor of Hawaii declared February 13 forever "Andy Irons Day".

 He is the only surfer to have won a title at every venue on the ASP calendar.
</item 6>

<item 7>
Triple Crown of Surfing
The Vans Triple Crown of Surfing is a Hawaiian specialty series of professional surfing events, offering three events to men and three events to women.

 For the men, those events are the Reef Hawaiian Pro at Haleiwa Ali'i Beach Park; the O'Neill World Cup of Surfing at Sunset Beach; and the Billabong Pipeline
Masters at the Banzai Pipeline.

 The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
</item 7>

<item 8>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 8>

<item 9>
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 9>

<item 10>
1981 Pepsi Grand Slam
The 1981 Pepsi Grand Slam was a men's tennis tournament played on outdoor clay courts in Boca Raton, Florida, USA The event was part of the 1981 Volvo Grand
Prix circuit.

 It was the sixth edition of the tournament and was held from February 14 through February 15, 1981.

 Four–time winner and defending champion Björn Borg withdrew a day before the tournament began after coming down with the flu.

 He was replaced by Vitas Gerulaitis.

 John McEnroe won the singles title and $150,000 first prize money.
</item 10>
=== END ITEM id=63300f3900f6b229 ===
