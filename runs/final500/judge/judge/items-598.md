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

=== ITEM id=781f5184e9c8dbe1 ===
Gold sentences and facts:
Sentence 0: Rivermaya is a Filipino rock band.
  Fact 0.0: Rivermaya is a Filipino rock band.
Sentence 1: Kodaline ( ) are an Irish band.
  Fact 1.0: Kodaline is an Irish band.

Retrieved items:
<item 1>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 1>

<item 2>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 2>

<item 3>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 3>

<item 4>
Kodaline
Kodaline ( ) are an Irish band.

 Originally known as 21 Demands, in 2012 the band changed their name to Kodaline.
</item 4>

<item 5>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 5>

<item 6>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 6>

<item 7>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 7>

<item 8>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 8>

<item 9>
Honest (Kodaline song)
"Honest" is a song by Dublin-based alternative rock quartet Kodaline.

 The song was released on 1 February 2015 as the lead single from the band's second studio album, "Coming Up for Air" (2015).

 "Honest" became the band's third top 10 single in their home country, Ireland, following "High Hopes" (2013) and "Love Like This" (2013).
</item 9>

<item 10>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 10>
=== END ITEM id=781f5184e9c8dbe1 ===

=== ITEM id=5ee95a7afa342dc0 ===
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
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 1>

<item 2>
Jeffrey R. MacDonald
Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in
February 1970.
</item 2>

<item 3>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 3>

<item 4>
Fatal Vision controversy
The controversy over Fatal Vision, journalist and author Joe McGinniss's best-selling 1983 true crime book, is a decades-long dispute spanning several court
cases and discussed in several other published works.
</item 4>

<item 5>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 5>

<item 6>
Fatal Vision (miniseries)
Fatal Vision is a 1984 American television miniseries based on the account, in the book of the same name, of the murders in 1970 at Fort Bragg of the wife and
daughters of U.S. Army officer Jeffrey R. MacDonald.
</item 6>

<item 7>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 7>

<item 8>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 8>

<item 9>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 9>

<item 10>
Fatal Vision (disambiguation)
Fatal Vision is the 1983 true crime book by Joe McGinniss which lies at the center of the "Fatal Vision" controversy.
</item 10>
=== END ITEM id=5ee95a7afa342dc0 ===

=== ITEM id=5881ad377b1fb30b ===
Gold sentences and facts:
Sentence 0:  He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems."
  Fact 0.0: The recipient was awarded the 2012 Nobel Prize in Physics.
  Fact 0.1: The recipient shared the 2012 Nobel Prize in Physics jointly with Serge Haroche.
  Fact 0.2: The 2012 Nobel Prize in Physics was awarded for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
Sentence 1:  Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.
  Fact 1.0: Since 2001, Serge Haroche has been a Professor at the Collège de France.
  Fact 1.1: Since 2001, Serge Haroche has held the Chair of Quantum Physics.

Retrieved items:
<item 1>
David J. Wineland
David Jeffrey Wineland (born February 24, 1944) is an American Nobel-laureate physicist at the National Institute of Standards and Technology (NIST) physics
laboratory.

 His work has included advances in optics, specifically laser cooling trapped ions and using ions for quantum computing operations.

 He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of
individual quantum systems."
</item 1>

<item 2>
List of Nobel laureates in Chemistry
The Nobel Prize in Chemistry (Swedish: "Nobelpriset i kemi" ) is awarded annually by the Royal Swedish Academy of Sciences to scientists in the various fields
of chemistry.

 It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, who died in 1896.

 These prizes are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.

 As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Royal
Swedish Academy of Sciences.

 The first Nobel Prize in Chemistry was awarded in 1901 to Jacobus Henricus van 't Hoff, of the Netherlands.

 Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.

 In 1901, van 't Hoff received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 2>

<item 3>
Brian Josephson
Brian David Josephson, {'1': ", '2': ", '3': ", '4': "} (born 4 January 1940), is a Welsh theoretical physicist and professor emeritus of physics at the
University of Cambridge.

 Best known for his pioneering work on superconductivity and quantum tunnelling, he was awarded the Nobel Prize in Physics in 1973 for his prediction of the
Josephson effect, made in 1962 when he was a 22-year-old PhD student at Cambridge University.

 Josephson is the only Welshman to have won a Nobel Prize in Physics.

 He shared the prize with physicists Leo Esaki and Ivar Giaever, who jointly received half the award for their own work on quantum tunnelling.
</item 3>

<item 4>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 4>

<item 5>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 5>

<item 6>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 6>

<item 7>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 7>

<item 8>
List of Nobel laureates in Physiology or Medicine
Each recipient receives a medal, a diploma and a monetary award that has varied throughout the years.

 In 1901, von Behring received 150,782 SEK, which is equal to 7,731,004 SEK in December 2008.

 In 2013, the prize was awarded to James E. Rothman, Randy W. Schekman and Thomas C. Südhof; they were recognised "after discovering how cells precisely
transport material".

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 8>

<item 9>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 9>

<item 10>
Nobel Prize in Physics
The Nobel Prize in Physics (Swedish: "Nobelpriset i fysik" ) is a yearly award given by the Royal Swedish Academy of Sciences for those who conferred the most
outstanding contributions for mankind in the field of physics.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel
Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 10>
=== END ITEM id=5881ad377b1fb30b ===

=== ITEM id=1855057affa8c3fb ===
Gold sentences and facts:
Sentence 0: Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.
  Fact 0.0: Emmitt H. Ford was born on December 13, 1943.
  Fact 0.1: Emmitt H. Ford died on November 10, 2014.
  Fact 0.2: Emmitt H. Ford was an American politician.
  Fact 0.3: Emmitt H. Ford was from Tennessee.
Sentence 1:  He was also an uncle of former United States Congressman Harold Ford Jr.
  Fact 1.0: Emmitt H. Ford was an uncle of Harold Ford Jr.
  Fact 1.1: Harold Ford Jr. is a former United States Congressman.
Sentence 2:  He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.
  Fact 2.0: Emmitt H. Ford is the son of Harold Ford Sr.
  Fact 2.1: Harold Ford Sr. is a former Congressman.
  Fact 2.2: Harold Ford Sr. held the same seat for 22 years.

Retrieved items:
<item 1>
Emmitt Ford
Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.

 He represented the 86th district encompassing Shelby County from 1975–1981.

 He was also an uncle of former United States Congressman Harold Ford Jr.
</item 1>

<item 2>
John Ford (Tennessee politician)
John N. Ford (born May 3, 1942 in Memphis, Tennessee), is a former Democratic member of the Tennessee State Senate and a member of Tennessee's most prominent
African-American political family.

 He is the older brother of former U.S. Representative Harold Ford, Sr. and the uncle of former Tennessee U.S. Representative and 2006 United States Senate
candidate Harold Ford, Jr. He is a member of the Ford political family from Memphis.
</item 2>

<item 3>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 3>

<item 4>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 4>

<item 5>
L. R. Ford Jr.
Lester Randolph Ford Jr. (born September 23, 1927 – February 26, 2017) was an American mathematician specializing in network flow problems.

 He was the son of mathematician Lester R. Ford Sr.
</item 5>

<item 6>
United States Senate election in Tennessee, 2006
The 2006 United States Senate election in Tennessee was held on November 7, 2006.

 The election winner, Bob Corker, served his term between January 3, 2007 and January 3, 2013.

 Corker replaced Republican Bill Frist in the Senate, who retired upon the end of his second term in 2007.

 Corker was the Republican nominee, and the Democratic nominee was Harold Ford, Jr..

 The race between Ford and Corker was one of the most competitive Senate races of 2006, with Corker winning the race by less than three percent of the vote.

 Corker was the only non-incumbent Republican to win a U.S. Senate seat in 2006.

 Since 1994, the Republican Party has held both of Tennessee's Senate seats.
</item 6>

<item 7>
Doug Ford Jr.
Douglas Bruce Ford Jr. (born November 20, 1964) is a Canadian businessman and politician in Toronto, Ontario.

 Ford was Toronto City Councillor for Ward 2 Etobicoke North in Toronto from 2010 to 2014 at the same time that his brother, Rob Ford (1969–2016), was mayor of
Toronto.

 Their father, Doug Ford Sr., was a Member of Provincial Parliament from 1995 to 1999 and founded Deco Labels and Tags, a printing business operating in Canada
and the United States.

 The business is now owned by Doug Ford Jr., who has served as company president since 2002, and his brother, Randy.
</item 7>

<item 8>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 8>

<item 9>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 9>

<item 10>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 10>
=== END ITEM id=1855057affa8c3fb ===

=== ITEM id=9e3ba97b95bb3498 ===
Gold sentences and facts:
Sentence 0: The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.
  Fact 0.0: The soundtrack for the 1968 film "Romeo and Juliet" was composed by Nino Rota.
  Fact 0.1: The soundtrack for the 1968 film "Romeo and Juliet" was conducted by Nino Rota.
Sentence 1: Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
  Fact 1.0: Romeo and Juliet is a 1968 British-Italian romance film.
  Fact 1.1: Romeo and Juliet is based on the tragic play of the same name.
  Fact 1.2: The play Romeo and Juliet was written between 1591 and 1595.
  Fact 1.3: The play Romeo and Juliet was written by William Shakespeare.

Retrieved items:
<item 1>
8½
8½ (Italian title: Otto e mezzo ] ) is a 1963 semi-autobiographical comedy-drama film directed by Federico Fellini.

 Co-scripted by Fellini, Tullio Pinelli, Ennio Flaiano, and Brunello Rondi, it stars Marcello Mastroianni as Guido Anselmi, a famous Italian film director.

 Shot in black-and-white by cinematographer Gianni di Venanzo, the film features a soundtrack by Nino Rota with costume and set designs by Piero Gherardi.
</item 1>

<item 2>
West Side Story (film)
West Side Story is a 1961 American musical romantic drama film directed by Robert Wise and Jerome Robbins.

 The film is an adaptation of the 1957 Broadway musical of the same name, which in turn was inspired by William Shakespeare's play "Romeo and Juliet".

 It stars Natalie Wood, Richard Beymer, Russ Tamblyn, Rita Moreno, and George Chakiris, and was photographed by Daniel L. Fapp, A.S.C., in Super Panavision 70.

 Released on October 18, 1961 through United Artists, the film received high praise from critics and viewers, and became the second highest grossing film of the
year in the United States.

 The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the
most wins for a movie musical.
</item 2>

<item 3>
The Ruffian on the Stair
The Ruffian On the Stair is a play by British playwright Joe Orton which was first broadcast on BBC Radio in August 1964.

 It is an unsympathetic yet comedic one-act portrayal of working class England, as played out by a couple and a mysterious young man who toys with their lives.

 It was based on "The Boy Hairdresser", a novel by Orton and his lover Kenneth Halliwell.

 The title and play are based on a few lines from poet and dramatist William Ernest Henley: "Madam Life's a piece in bloom, / Death goes dogging everywhere: /
She's the tenant of the room, / He's the ruffian on the stair."

 "Ruffian" is not as renowned as other works such as "Loot" and "What the Butler Saw", but it is still staged on occasion.
</item 3>

<item 4>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 4>

<item 5>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 5>

<item 6>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 6>

<item 7>
Il cappello di paglia di Firenze
Il cappello di paglia di Firenze (literally translated as "The Florentine Straw Hat" but usually titled in English language productions as "The Italian Straw
Hat") is an opera by Nino Rota to an Italian-language libretto by the composer and Ernesta Rota, based on the play "Le chapeau de paille d'Italie" by Eugène
Labiche and Marc Michel.
</item 7>

<item 8>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 8>

<item 9>
Romeo and Juliet (1968 film)
Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
</item 9>

<item 10>
Amarcord Nino Rota
Amarcord Nino Rota is an album by various artists, recorded as a tribute to composer Nino Rota.
</item 10>
=== END ITEM id=9e3ba97b95bb3498 ===

=== ITEM id=d85d0f77b37a55d1 ===
Gold sentences and facts:
Sentence 0: Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National Remembrance.
  Fact 0.0: Kolejka is a board game.
  Fact 0.1: Kolejka is Polish for 'queue' or 'line (of people)'.
  Fact 0.2: Karol Madaj designed Kolejka.
  Fact 0.3: Kolejka was launched in February 2011.
  Fact 0.4: Kolejka was launched by Poland's Institute of National Remembrance.
Sentence 1: Ghettopoly is a parody of "Monopoly" released in 2003.
  Fact 1.0: Ghettopoly is a parody of Monopoly.
  Fact 1.1: Ghettopoly was released in 2003.

Retrieved items:
<item 1>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 1>

<item 2>
Sarra Manning
Her first adult novel, "Unsticky" was published by Headline in 2009.

 Her next teen novel, "Nobody's Girl" was published in 2010, and a second adult novel, "You Don't Have to Say You Love Me", was published in February 2011.

 Her third adult novel, titled "Nine Uses for an Ex-Boyfriend", was published in February 2012.
</item 2>

<item 3>
The White EP (Vib Gyor EP)
The "White EP" is a promo EP from the Leeds band Vib Gyor.

 It was first sold on their UK tour with American band Transfer and was released through their My Space page and official web-site on 13 November 2007.

 It is the follow-up to their The Secret EP.

 It was recorded in Rockfield studios in South Wales and produced by Ken Thomas.
</item 3>

<item 4>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 4>

<item 5>
Kick Over the Traces
Kick Over the Traces is a compilation album by the American punk rock band Pinhead Gunpowder.

 The album contains tracks from Pinhead Gunpowder releases since the band formed in 1990.

 The album was released through Recess Records on June 16, 2009.

 The Japanese edition features a second disc recorded live at 924 Gilman Street on February 10, 2008, and features different cover art.
</item 5>

<item 6>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 6>

<item 7>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 7>

<item 8>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 8>

<item 9>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 9>

<item 10>
Ghettopoly
Ghettopoly is a parody of "Monopoly" released in 2003.

 Invented by David Chang, it uses "Monopoly"-like mechanics in the atmosphere of a caricaturized United States ghetto.
</item 10>
=== END ITEM id=d85d0f77b37a55d1 ===

=== ITEM id=30fdaadf7ae093f1 ===
Gold sentences and facts:
Sentence 0: "What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).
  Fact 0.0: What the Water Gave Me is a song
  Fact 0.1: What the Water Gave Me is by English indie rock band Florence and the Machine
  Fact 0.2: What the Water Gave Me is from the band's second studio album Ceremonials
  Fact 0.3: The album Ceremonials is dated 2011
Sentence 1:  The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.
  Fact 1.0: Lead singer Florence Welch wrote the song What the Water Gave Me
  Fact 1.1: Francis Eg White wrote the song What the Water Gave Me
  Fact 1.2: Paul Epworth produced the song What the Water Gave Me
Sentence 2: What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.
  Fact 2.0: What the Water Gave Me is an oil painting by Frida Kahlo
  Fact 2.1: The painting is titled Lo que el agua me dio in Spanish
  Fact 2.2: Frida Kahlo completed the painting in 1938

Retrieved items:
<item 1>
Spectrum (Say My Name)
"Spectrum" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 It was written by lead singer Florence Welch and Paul Epworth with production being handled by Epworth.

 The band premiered the song during a concert at Brooklyn's Creators Project on 15 October 2011, prior to the release of the album.

 The album version of "Spectrum" is a downtempo orchestral pop, R&B, krautrock and disco song.

 The accompanying music video for the song premiered on 30 May 2012.

 It was directed by David LaChapelle and John Byrne.
</item 1>

<item 2>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 2>

<item 3>
Never Let Me Go (Florence and the Machine song)
"Never Let Me Go" is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by Florence Welch and Paul Epworth while production was handled by Epworth.

 Island Records released the song as the third single from the album on 30 March 2012.

 The band first premiered the song during one of their concerts before the release of the album on 15 October 2011.

 A Clams Casino remix of the song was placed on the 12" white vinyl single which was available for pre-order on 22 March 2012.
</item 3>

<item 4>
Wish That You Were Here
"Wish That You Were Here" is a song by the English indie rock band Florence and the Machine, written by Florence Welch, Andrew Wyatt, and Emilie Haynie, and was
released on Island Records on 26 August 2016.

 The song was made available via digital download, and is featured on the soundtrack of the film "Miss Peregrine's Home for Peculiar Children" which was
directed by Tim Burton.

 Welch — a long-time fan of Burton's work — had expressed interest in collaborating with him long before the recording of "Wish That You Were Here" as they
shared similar artistic themes.

 Upon release, the composition was positively received by music critics and reached number 128 on the UK Singles Chart.
</item 4>

<item 5>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 5>

<item 6>
What the Water Gave Me (painting)
What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.

 It is sometimes referred to as "What I Saw in the Water".
</item 6>

<item 7>
Shake It Out
"Shake It Out" is a song by English indie rock band Florence and the Machine, released as the first official single from their second studio album,
"Ceremonials" (2011).

 It was written by Florence Welch and Paul Epworth, while production was handled by Epworth.

 The song was digitally released in Australia on 14 September 2011, and it was available in the United States on 19 October.

 It had its radio debut on XFM on 14 September 2011 in the United Kingdom.

 Welch revealed that the song was written within an hour and according to her it talked about shaking the regrets and the things that were haunting her.
</item 7>

<item 8>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 8>

<item 9>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 9>

<item 10>
Lover to Lover
"Lover to Lover" is a song by English indie rock band Florence and the Machine.

 The song was released on 30 November 2012, as a digital download on iTunes as the fifth and final single from their second studio album "Ceremonials" (2011).

 The song was written by Florence Welch and Francis "Eg" White and produced by Paul Epworth.

 The Ceremonials Tour version of "Lover to Lover" is an edited version of the original with further production by Chris Hayden.

 The song became the band's first single to miss the top 200 in the UK Singles Chart.
</item 10>
=== END ITEM id=30fdaadf7ae093f1 ===

=== ITEM id=0a072abbe53eca41 ===
Gold sentences and facts:
Sentence 0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.1: Alberta is a Canadian province.
  Fact 0.2: Alberta is governed under fundamental rules described in the Constitution of Alberta.
Sentence 1:  As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.
  Fact 1.0: Alberta's constitution is an unwritten constitution.
  Fact 1.1: It is typical of all Canadian provinces for their constitutions to be unwritten.
  Fact 1.2: It is typical of Westminster systems more generally for their constitutions to be unwritten.
Sentence 2: The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.
  Fact 2.0: The Westminster system is a parliamentary system of government.
  Fact 2.1: The Westminster system is modelled after the system of government that developed in the United Kingdom.

Retrieved items:
<item 1>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 1>

<item 2>
Quebec Legislature
The Quebec Legislature (officially Parliament of Quebec, French: "Parlement du Québec" ) is the legislature of the province of Quebec, Canada, since
Confederation in 1867.

 Today, the legislature is made of two elements: the monarch of Canada, represented by the Lieutenant Governor of Quebec, and the unicameral assembly called the
National Assembly of Quebec.

 The legislature has existed since 1867 when Quebec, then called Canada East, became one of the founding colonies of Canadian Confederation.

 From 1867 until 1968 the legislature was bicameral, containing a lower chamber called the Legislative Assembly of Quebec and an upper chamber called the
Legislative Council of Quebec.
</item 2>

<item 3>
Charter colony
Charter colony is one of three classes of colonial government established in the 17th century English colonies in North America, the other classes being
proprietary colony and royal colony.

 The colonies of Rhode Island, Connecticut, and Massachusetts Bay were charter colonies.

 In a charter colony, Britain granted a charter to the colonial government establishing the rules under which the colony was to be governed.

 The charters of Rhode Island and Connecticut granted the colonists significantly more political liberty than other colonies.

 Rhode Island and Connecticut continued to use their colonial charters as their State constitutions after the American Revolution.
</item 3>

<item 4>
Proprietary colony
A proprietary colony was a type of British colony mostly in North America and the Caribbean in the 17th century.

 In the British Empire, all land belonged to the ruler, and it was his prerogative to divide.

 Therefore, all colonial properties were partitioned by royal charter into one of four types: proprietary, royal, joint stock, or covenant.

 King Charles II used the proprietary solution to reward allies and focus his own attention on Britain itself.

 He offered his friends colonial charters which facilitated private investment and colonial self-government.

 The charters made the proprietor the effective ruler, albeit one ultimately responsible to English law and the king.

 Charles II gave New Netherland to his younger brother The Duke of York, who named it New York.

 He gave an area to William Penn who named it Pennsylvania.
</item 4>

<item 5>
Westminster system
The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.

 This term comes from the Palace of Westminster, the seat of the British parliament.
</item 5>

<item 6>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 6>

<item 7>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 7>

<item 8>
Constitution of Alberta
The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.

 As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.

 Alberta's constitution, like Britain's (on which it is modeled), includes any and all pieces of legislation, court decisions, proclamations, and conventions
which together inform how the province operates.

 Many statutes are important to understanding the governance of the province, but nowhere are they consolidated into a single document or even a list.

 The office of Attorney-General at one time suggested 23 acts which might be included, but cautioned that this was not a "definitive list".

 However, since Alberta is a part of federation, its powers are clearly delineated in law, via the Constitution of Canada.
</item 8>

<item 9>
British North America Acts
The British North America Acts 1867–1975 are the original names of a series of Acts at the core of the constitution of Canada.

 They were enacted by the Parliament of the United Kingdom and the Parliament of Canada.

 In Canada, some of the Acts were amended or repealed by the Constitution Act, 1982.

 The rest were renamed in Canada as the "Constitution Acts".

 In the United Kingdom, those Acts that were passed by the British Parliament remain under their original names.

 The term "British North America" (BNA) refers to the British colonies in North America.
</item 9>

<item 10>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 10>
=== END ITEM id=0a072abbe53eca41 ===

=== ITEM id=2efb993692a7cc79 ===
Gold sentences and facts:
Sentence 0: Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
  Fact 0.0: Ernst Willi Messerschmid is Prof. Dr.
  Fact 0.1: Ernst Willi Messerschmid was born on May 21, 1945.
  Fact 0.2: Ernst Willi Messerschmid is a German physicist.
  Fact 0.3: Ernst Willi Messerschmid is a former astronaut.
Sentence 1: Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
  Fact 1.0: Philippe Perrin was born on January 6, 1963.
  Fact 1.1: Philippe Perrin is a Colonel in the French Air Force.
  Fact 1.2: Philippe Perrin is a French test pilot.
  Fact 1.3: Philippe Perrin is a former CNES astronaut.
  Fact 1.4: Philippe Perrin is a former European Space Agency astronaut.

Retrieved items:
<item 1>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 1>

<item 2>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 2>

<item 3>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 3>

<item 4>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 4>

<item 5>
Philippe Perrin (artist)
Philippe Perrin, (La Tronche, 10 August 1964) is a French artist and contemporary photographer who lives and works in Paris.
</item 5>

<item 6>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 6>

<item 7>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 7>

<item 8>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 8>

<item 9>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 9>

<item 10>
New Chronology (Fomenko)
The New Chronology is a pseudohistorical theory which argues that the conventional chronology of Middle Eastern and European history is fundamentally flawed,
and that events attributed to the civilizations of the Roman Empire, Ancient Greece and Ancient Egypt actually occurred during the Middle Ages, more than a
thousand years later.

 The central concepts of the New Chronology are derived from the ideas of Russian scholar Nikolai Morozov (1854–1946), although work by French scholar Jean
Hardouin (1646–1729) can be viewed as an earlier predecessor.

 However, the New Chronology is most commonly associated with Russian mathematician Anatoly Fomenko (born 1945), although published works on the subject are
actually a collaboration between Fomenko and several other mathematicians.

 The concept is most fully explained in "History: Fiction or Science?"

, originally published in Russian.
</item 10>
=== END ITEM id=2efb993692a7cc79 ===

=== ITEM id=3b88b9949d8eb67a ===
Gold sentences and facts:
Sentence 0: Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.
  Fact 0.0: Mirrors is a 2008 supernatural horror film.
  Fact 0.1: Mirrors was directed by Alexandre Aja.
  Fact 0.2: Kiefer Sutherland starred in Mirrors.
  Fact 0.3: Paula Patton starred in Mirrors.
  Fact 0.4: Amy Smart starred in Mirrors.
Sentence 1: Paula Maxine Patton (born December 5, 1975) is an American actress.
  Fact 1.0: Paula Maxine Patton was born on December 5, 1975.
  Fact 1.1: Paula Maxine Patton is an American actress.

Retrieved items:
<item 1>
Mirrors (film)
Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.

 The film was first titled "Into the Mirror", but the name was later changed to "Mirrors".

 Filming began on May 1, 2007, and it was released in American theaters on August 15, 2008.
</item 1>

<item 2>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 2>

<item 3>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 3>

<item 4>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 4>

<item 5>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 5>

<item 6>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 6>

<item 7>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 7>

<item 8>
Paula Patton
Paula Maxine Patton (born December 5, 1975) is an American actress.

 Patton made her film debut in the 2005 comedy film "Hitch" and starred in the epic fantasy film "Warcraft" (2016), based on the game series of the same name.

 Patton has also been the female lead in "Déjà Vu" (2006), "" (2011) and "2 Guns" (2013), and appeared in the critically acclaimed "Precious" (2009).
</item 8>

<item 9>
Hitch (film)
Hitch is a 2005 American romantic comedy film directed by Andy Tennant and starring Will Smith.

 The film, which was written by Kevin Bisch, co-stars Eva Mendes, Kevin James, and Amber Valletta.

 Smith plays the main fictional character of the film, Alex "Hitch" Hitchens, who is a professional dating consultant who makes a living teaching men how to woo
women.

 The film was released on February 11, 2005 by Columbia Pictures.
</item 9>

<item 10>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 10>
=== END ITEM id=3b88b9949d8eb67a ===

=== ITEM id=4f3cbc4150ca3d59 ===
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
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 1>

<item 2>
Barry Ackerley headed Ackerley Group and owned the Seattle SuperSonics (1983–2001); he died after a stroke in 2011. Al Carter was an award-winning
Texas/Oklahoma sportswriter. André Laguerre led Sports Illustrated (1960–74) and introduced its Swimsuit Issue. Dick “Scoop” Gordon was a longtime sports
journalist. Cork fiddler John Murphy learned from his father. Galveston gang leader Johnny Jack N
</item 2>

<item 3>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 3>

<item 4>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
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
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 6>

<item 7>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 7>

<item 8>
This Is Bing Crosby
This Is Bing Crosby was a fifteen-minute five times a week daytime radio program featuring Bing Crosby acting as a disc jockey.

 Minute Maid quick frozen concentrated orange juice was promoted on the shows.
</item 8>

<item 9>
Dick Gordon (sports writer)
Charles Richards Gordon, known as Dick "Scoop" Gordon (January 15, 1911 – December 8, 2008), was an American sports journalist whose works were a regular
feature in venerable sports magazines like "The Sporting News", "Sports Illustrated", and "Baseball Digest".

 After earning his nickname "Scoop" in 1930 by reporting for "The Daily Princetonian" that golfing legend Bobby Jones would be retiring from active competition,
Gordon went on to a sports reporting career which ended in 2008.
</item 9>

<item 10>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 10>
=== END ITEM id=4f3cbc4150ca3d59 ===

=== ITEM id=e7a45686fbaec5d4 ===
Gold sentences and facts:
Sentence 0: Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.
  Fact 0.0: Isaac Asimov was born Isaak Ozimov.
  Fact 0.1: Isaac Asimov was born on January 2, 1920.
  Fact 0.2: Isaac Asimov's dates run from January 2, 1920 to April 6, 1992.
  Fact 0.3: Isaac Asimov was a Russian-American writer.
  Fact 0.4: Isaac Asimov was a professor of biochemistry at Boston University.
Sentence 1: Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.
  Fact 1.0: Michael Cunningham was born on November 6, 1952.
  Fact 1.1: Michael Cunningham is a U.S. novelist.
  Fact 1.2: Michael Cunningham is a U.S. screenwriter.

Retrieved items:
<item 1>
The entries cover magic effects; Harry Potter creatures, many drawn from folklore; Karen Joy Fowler’s Nebula-winning story “Always”; and the no-fee Writers of
the Future contest. Several focus on Isaac Asimov: his prolific career and “The Last Question”; the Robot City and Robots and Aliens novel series; and his
twelve-volume Magical Worlds of Fantasy anthologies, including “Witches” and “Wizards.” Other anthologies include Jewish science fiction collection “Wandering
Stars”; “
</item 1>

<item 2>
Isaac Asimov's Robot City
Isaac Asimov's Robot City is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 It takes place between "The Robots of Dawn" and "Robots and Empire".

 Each volume is complete in itself, but they form a continuing series.

 The novels were written in response to a writing challenge issued by Asimov to write a series involving the Three Laws of Robotics, which brought about a
collaboration of several authors.

 Asimov provided outlines for stories which filled in the gap between Asimov's own robot stories and his "Foundation" series, explaining the disappearance of
the robots prior to the establishment of the galactic empire. "

Isaac Asimov's Robots and Aliens" followed in this series, with the same protagonists and many other characters.

 The common theme of all books of both series is the interaction between the characters and autonomous cities run and populated by robots (the "robot cities" of
the series title).

 Robot City was also released as a mystery game for the PC in 1995.

 The player takes the role of Derec.
</item 2>

<item 3>
Isaac Asimov's Robot City: Odyssey
Isaac Asimov's Robot City: Odyssey is a young adult science fiction novel written in 1987 by Michael P. Kube-McDowell.

 It is part of the series Isaac Asimov's Robot City, inspired by Isaac Asimov's "Robot" series.

 The 1995 computer game Robot City is based on the plot of "Odyssey".
</item 3>

<item 4>
The Last Question
"The Last Question" is a science fiction short story by American writer Isaac Asimov.

 It first appeared in the November 1956 issue of "Science Fiction Quarterly" and was anthologized in the collections "Nine Tomorrows" (1959), "The Best of Isaac
Asimov" (1973), "Robot Dreams" (1986), the retrospective "Opus 100" (1969), and in "Isaac Asimov: The Complete Stories, Vol.

 1" (1990).

 It was Asimov's favorite short story of his own authorship, and is one of a loosely connected series of stories concerning a fictional computer called
Multivac.

 The story overlaps science fiction, theology, and philosophy.
</item 4>

<item 5>
Michael Cunningham
Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.

 He is best known for his 1998 novel "The Hours", which won the Pulitzer Prize for Fiction and the PEN/Faulkner Award in 1999.

 Cunningham is a senior lecturer of creative writing at Yale University.
</item 5>

<item 6>
Isaac Asimov's Robot City: Perihelion
Isaac Asimov's Robot City: Perihelion is a book written in 1988 by William F. Wu.

 It is part of the series Isaac Asimov's Robot City, which was inspired by Isaac Asimov's "Robot" series.
</item 6>

<item 7>
Wizards (Asimov anthology)
Wizards is an anthology of themed fantasy and science fiction short stories on the subject of wizards edited by Isaac Asimov, Martin H. Greenberg and Charles G.
Waugh.

 The first volume in their Isaac Asimov's Magical Worlds of Fantasy series, it was first published in paperback by Signet/New American Library in October 1983.

 It was later gathered together with "Witches", the second book in the series, into the omnibus hardcover collection "Isaac Asimov's Magical Worlds of Fantasy:
Witches & Wizards" (1985).
</item 7>

<item 8>
Isaac Asimov
Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.

 He was known for his works of science fiction and popular science.

 Asimov was a prolific writer, and wrote or edited more than 500 books and an estimated 90,000 letters and postcards. His books have been published in 9 of the
10 major categories of the Dewey Decimal Classification.
</item 8>

<item 9>
Isaac Asimov's Robots and Aliens
Isaac Asimov's Robots and Aliens is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 Each volume is complete in itself, but they form a continuing series.

 The series follows the action of the novels of the "Isaac Asimov's Robot City" series, with the same protagonists Derec and Ariel, and many other characters.

 The plot deals with the Three Laws and encounters between robots and different varieties of alien life.
</item 9>

<item 10>
Isaac Asimov's Magical Worlds of Fantasy
Isaac Asimov's Magical Worlds of Fantasy is a series of twelve themed paperback fantasy and science fiction anthologies edited by Isaac Asimov, Martin H.
Greenberg and Charles G. Waugh, a companion set to the ten volume "Isaac Asimov's Wonderful Worlds of Science Fiction", produced by the same editors.

 It was published by Signet/New American Library from 1983 to 1991.

 Volumes 1 and 2 were also issued in hardcover in an omnibus collection titled "Isaac Asimov's Magical Worlds of Fantasy: Witches & Wizards".
</item 10>
=== END ITEM id=e7a45686fbaec5d4 ===
