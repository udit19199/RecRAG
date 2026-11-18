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

=== ITEM id=46a8af990cd5fddc ===
Gold sentences and facts:
Sentence 0: German Spitz is used to refer to both a breed of dog and category or type of dog.
  Fact 0.0: German Spitz is used to refer to a breed of dog.
  Fact 0.1: German Spitz is used to refer to a category or type of dog.
Sentence 1:  Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.
  Fact 1.0: Several modern breeds have been developed from the German Spitz.
  Fact 1.1: These modern breeds are either registered as separate breeds.
  Fact 1.2: These modern breeds are either registered as varieties of German Spitz.
Sentence 2:  The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that would later become known as the English Springer Spaniel.
  Fact 2.0: The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels.
  Fact 2.1: Attempts were made to use the term to specify a breed that would later become known as the English Springer Spaniel.

Retrieved items:
<item 1>
King Charles Spaniel
The King Charles Spaniel (also known as the English Toy Spaniel) is a small dog breed of the spaniel type.

 In 1903, the Kennel Club combined four separate toy spaniel breeds under this single title.

 The other varieties merged into this breed were the Blenheim, Ruby and Prince Charles Spaniels, each of which contributed one of the four colours available in
the breed.
</item 1>

<item 2>
American Cocker Spaniel
The American Cocker Spaniel is a breed of sporting dog.

 It is a spaniel type dog that is closely related to the English Cocker Spaniel; the two breeds diverged during the 20th century due to differing breed
standards in America and the UK.

 In the United States, the breed is usually called the Cocker Spaniel, while elsewhere in the world, it is called the American Cocker Spaniel in order to
differentiate between it and its English cousin, which was already known as "Cocker Spaniel" before the American variety was created.

 The word "cocker" is commonly held to stem from their use to hunt woodcock in England, while "spaniel" is thought to be derived from the type's origins in
Spain.
</item 2>

<item 3>
English Springer Spaniel
The English Springer Spaniel is a breed of gun dog in the Spaniel family traditionally used for flushing and retrieving game.

 It is an affectionate, excitable breed with a typical lifespan of twelve to fourteen years.

 They are very similar to the Welsh Springer Spaniel and are descended from the Norfolk or Shropshire Spaniels of the mid-19th century; the breed has diverged
into separate show and working lines.

 The breed suffers from average health complaints.

 The show-bred version of the breed has been linked to "rage syndrome", although the disorder is very rare.

 It is closely related to the Welsh Springer Spaniel and very closely to the English Cocker Spaniel; less than a century ago, springers and cockers would come
from the same litter.

 The smaller "cockers" hunted woodcock while the larger littermates were used to flush, or "spring," game.

 In 1902, The Kennel Club recognized the English Springer Spaniel as a distinct breed.

 They are used as sniffer dogs on a widespread basis.

 The term "Springer" comes from the historic hunting role, where the dog would flush (spring) birds into the air.
</item 3>

<item 4>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 4>

<item 5>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 5>

<item 6>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 6>

<item 7>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 7>

<item 8>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 8>

<item 9>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 9>

<item 10>
Norfolk Spaniel
The Norfolk Spaniel or Shropshire Spaniel is an extinct breed of dog since the early 20th century.

 It was originally thought to have originated from the work of one of the Dukes of Norfolk, but this theory was disproven after being in doubt during the later
part of the 19th century.

 The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that
would later become known as the English Springer Spaniel.
</item 10>
=== END ITEM id=46a8af990cd5fddc ===

=== ITEM id=f1aaba618cfaaf72 ===
Gold sentences and facts:
Sentence 0: The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.
  Fact 0.0: The 2012 United States Libertarian National Convention was a convention of the Libertarian Party (LP).
  Fact 0.1: Delegates of the Libertarian Party chose the party's nominees at the 2012 United States Libertarian National Convention.
  Fact 0.2: The nominees chosen at the 2012 United States Libertarian National Convention were for President of the United States.
  Fact 0.3: The nominees chosen at the 2012 United States Libertarian National Convention were for Vice President of the United States.
  Fact 0.4: The nominees were chosen for the 2012 general election.
  Fact 0.5: The 2012 United States Libertarian National Convention was held May 2-6, 2012.
  Fact 0.6: The 2012 United States Libertarian National Convention was held in Las Vegas, Nevada.
  Fact 0.7: The 2012 United States Libertarian National Convention was held at the Red Rock Resort Spa and Casino.
Sentence 1:  Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.
  Fact 1.0: Gary Johnson is a former Governor of New Mexico.
  Fact 1.1: Gary Johnson won the presidential nomination.
  Fact 1.2: Gary Johnson won the presidential nomination on the first ballot.
Sentence 2: Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.
  Fact 2.0: Gary Earl Johnson was born on January 1, 1953.
  Fact 2.1: Gary Earl Johnson is an American businessman.
  Fact 2.2: Gary Earl Johnson is an American author.
  Fact 2.3: Gary Earl Johnson is an American politician.
Sentence 3:  He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.
  Fact 3.0: Gary Johnson was the 29th Governor of New Mexico.
  Fact 3.1: Gary Johnson was Governor of New Mexico from 1995 to 2003.
  Fact 3.2: Gary Johnson was a member of the Republican Party as Governor of New Mexico.

Retrieved items:
<item 1>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 1>

<item 2>
Libertarian Party presidential primaries, 2016
The 2016 Libertarian Party presidential primaries and caucuses allowed electors to indicate non-binding preferences for the Libertarian Party's presidential
candidate.

 These differed from the Republican or Democratic presidential primaries and caucuses in that they did not appoint delegates to represent a candidate at the
party's convention to select the party's nominee for the United States presidential election.

 The party's nominee for the 2016 presidential election was chosen directly by registered delegates at the 2016 Libertarian National Convention, which ran from
May 26 to 30, 2016.

 The delegates nominated former New Mexico Governor Gary Johnson and former Massachusetts Governor Bill Weld.
</item 2>

<item 3>
2012 Libertarian National Convention
The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United
States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.

 Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.

 Retired California state court judge Jim Gray won the vice-presidential nomination, also on the first ballot.

 The convention also chose to replace most of the Libertarian National Committee party officers and members-at-large.
</item 3>

<item 4>
Libertarian National Committee
The Libertarian National Committee (LNC) controls and manages the affairs, properties, and funds of the United States Libertarian Party.

 It is composed of the party officers, five at-large representatives elected every two years at the national convention, and a theoretical maximum of 10
regional representatives.

 The current chair, elected in June 2014, is Nicholas Sarwark.

 The Executive Director is Wes Benedict.
</item 4>

<item 5>
Gary Johnson presidential campaign, 2012
The 2012 presidential campaign of Gary Johnson, the 29th Governor of New Mexico, was announced on April 21, 2011.

 He declared his candidacy for the 2012 Republican Party nomination for President of the United States.

 On December 28, 2011, Johnson withdrew his candidacy for the Republican nomination, and declared his candidacy for the 2012 presidential nomination of the
Libertarian Party.

 The 2012 Libertarian National Convention was held during the first weekend of May 2012.

 On May 5, 2012, after promoting his libertarian-oriented political positions to delegates, Johnson received the most votes at the convention and became the
official 2012 Libertarian presidential nominee.

 On November 6, 2012, Johnson received just under 1% of the popular vote in the general election, amounting to more than 1.2 million votes, more than double
what the Barr/Root ticket received in 2008.

 This was the most successful result for a third-party presidential candidacy since 2000, and the best in the Libertarian Party's history by vote number at the
time.

 (Johnson ran again in 2016 and received nearly four times his 2012 vote total.)
</item 5>

<item 6>
Political positions of Gary Johnson
Gary Johnson was the governor of New Mexico from 1995 to 2003 and ran for president in 2012 and 2016.

 In December 2011 he announced he would pursue the presidential nomination of the Libertarian Party for the 2012 general election.

 The Libertarian National Convention in May, 2012 chose Johnson as the party's candidate.

 In November 2014, Johnson announced he would pursue the presidential nomination of the Libertarian Party for the 2016 general election.

 Johnson has taken positions on many political issues as revealed through his public comments, his gubernatorial record, his Our America Initiative and his
campaigns to win the Republican and Libertarian nominations.
</item 6>

<item 7>
Gary Johnson
Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.

 He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.

 He was also the Libertarian Party nominee for President of the United States in the 2012 and 2016 elections.
</item 7>

<item 8>
David Bergland
David Peter Bergland (born June 4, 1935) is an American politician who was the United States Libertarian Party's nominee at the 1983 Libertarian National
Convention for President of the United States in the 1984 presidential election.
</item 8>

<item 9>
Wayne W. Lambert
The Brigadier General retired from the US Air Force in 1989.

 Lambert also holds an MBA from Chapman College (’76).

 In retirement, Lambert publicly supported U.S. presidential candidate Mitt Romney and U.S. vice presidential candidate Paul Ryan in the 2012 general election.
</item 9>

<item 10>
Libertarian National Convention
The Libertarian National Convention is held every two years by the Libertarian Party (United States) to choose members of the Libertarian National Committee
(LNC), and to conduct other party business.

 In presidential election years, the convention delegates enact a platform and nominate the Libertarian presidential and vice-presidential candidates who then
face the nominees of other parties in the November general election.
</item 10>
=== END ITEM id=f1aaba618cfaaf72 ===

=== ITEM id=e6a105f687816af2 ===
Gold sentences and facts:
Sentence 0: Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.
  Fact 0.0: Bumble and bumble Products, LLC is one of 27 brands.
  Fact 0.1: The 27 brands are owned by Estée Lauder Companies, Inc.
  Fact 0.2: Bumble and bumble Products, LLC is owned by Estée Lauder Companies, Inc.
Sentence 1:  The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown Manhattan, New York City.
  Fact 1.0: Estée Lauder Companies, Inc. owns a diverse portfolio of brands.
  Fact 1.1: The brand portfolio is distributed internationally through retail channels.
  Fact 1.2: The brand portfolio is distributed internationally through digital commerce.
  Fact 1.3: Estée Lauder Companies, Inc. has its headquarters in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 1>

<item 2>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 2>

<item 3>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 3>

<item 4>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 4>

<item 5>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 5>

<item 6>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 6>

<item 7>
A. H. Belo
A. H. Belo Corporation is a Dallas-based media company that owns newspapers in North Texas.

 The current corporation was formed when Belo Corporation separated its broadcasting and publishing operations into two corporations.

 A. H. Belo also owns a part interest in Classified Ventures.

 Its CEO is Robert Decherd who had headed Belo before the spin-off.

 The company has its headquarters in the Belo Building in Downtown Dallas.
</item 7>

<item 8>
Rubin Schron
Rubin Schron, who goes by Ruby, is a New York City real estate investor, landlord and the founder of Cammeby's International Group.

 He has a personal net worth of well over $10 billion, according to data company Real Capital Analytics.

 The portfolio of Cammeby's, which Schron founded in 1967, includes office buildings, market-rate and government-subsidized apartment complexes, nursing homes,
the 16-building complex in Sunset Park now known as Industry City, a stake in the bottom half of Woolworth Building and industrial properties scattered across
Long Island.

 In 2013, Schron made an unsolicited and unsuccessful offer to buy the Empire State Building for $2 billion, but he has not had problems closing many other
deals throughout his long career.

 In 2003, an investment group led by Schron paid $600 million for a portfolio of about 6,000 outer-borough apartments from Donald Trump.
</item 8>

<item 9>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 9>

<item 10>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 10>
=== END ITEM id=e6a105f687816af2 ===

=== ITEM id=8d8ec917a7df25de ===
Gold sentences and facts:
Sentence 0: Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.
  Fact 0.0: Jelena Janković was born on 28 February 1985.
  Fact 0.1: Jelena Janković's name is written in Serbian Cyrillic as Јелена Јанковић.
  Fact 0.2: Jelena Janković is a Serbian professional tennis player.
Sentence 1: Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian politician and retired professional tennis player.
  Fact 1.0: Marat Mubinovich Safin's name is written in Russian as Мара́т Муби́нович Са́фин.
  Fact 1.1: Marat Mubinovich Safin's name is written in Tatar as Марат Мөбин улы Сафин.
  Fact 1.2: Marat Mubinovich Safin's Tatar name is romanized as Marat Möbin uğlı Safin.
  Fact 1.3: Marat Mubinovich Safin was born on 27 January 1980.
  Fact 1.4: Marat Mubinovich Safin is a Russian politician.
  Fact 1.5: Marat Mubinovich Safin is a retired professional tennis player.

Retrieved items:
<item 1>
Ivanovic–Janković rivalry
Serbian tennis players Ana Ivanovic and Jelena Janković are both former world No. 1 and have met 12 times so far.

 Their overall head-to-head series is 9–3, in Ivanovic's favor.

 Janković is two years older than Ivanovic, so their paths connect them to play each other.

 They are the first female players to achieve major successes for independent Serbia.
</item 1>

<item 2>
Jelena Janković
Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.

 Janković is a former World No. 1 in singles, a ranking achieved preceding her finals appearance at the 2008 US Open.

 Janković's career highlights include winning the 2007 Wimbledon Mixed Doubles title with Jamie Murray, the 2010 Indian Wells Masters, the Internazionali BNL
d'Italia twice, in 2007 and 2008, and the aforementioned appearance in the finals of the 2008 US Open.

 She is coached by her brother Marko.
</item 2>

<item 3>
Ana Ivanovic career statistics
This is a list of the main career statistics of Serbian professional tennis player, Ana Ivanovic.

 Ivanovic won fifteen WTA singles titles including one grand slam singles title at the 2008 French Open and three WTA Tier I singles titles.

 She was also the runner-up at the 2007 French Open and 2008 Australian Open and a semi-finalist at the 2007 Wimbledon Championships and 2007 WTA Tour
Championships.

 On June 9, 2008 Ivanovic became the world No. 1 for the first time in her career.
</item 3>

<item 4>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 4>

<item 5>
2008 Wimbledon Championships – Mixed Doubles
Jamie Murray and Jelena Janković were the defending champions, but Janković chose not to participate, and only Murray competed that year.
</item 5>

<item 6>
Jelenin svet
Jelenin svet ("Jelena's World") is a 2008 independent documentary film written and directed by Tanja Brzaković, about former World No. 1 female tennis player,
Jelena Janković.
</item 6>

<item 7>
Graph query row:
person1: Jelena Janković
person1_occupation: professional tennis player
person1_nationality: Serbian
person2: Marat Safin
person2_occupation: ['politician', 'retired professional tennis player']
person2_nationality: Russian
connecting_entity: None
relationship_from_person1: None
relationship_to_person2: None
</item 7>

<item 8>
Dinara Safina career statistics
This is a list of the main career statistics of retired, Russian professional tennis player, Dinara Safina.

 Throughout her career, Safina won twelve WTA singles titles including three Tier I singles titles at the 2008 Qatar Telecom German Open, Rogers Cup and Toray
Pan Pacific Open respectively; one Premier Mandatory singles title at the 2009 Mutua Madrileña Madrid Open and one Premier 5 singles title at the 2009
Internazionali BNL d'Italia.

 She was also the runner-up at the 2008 French Open and the 2009 Australian Open and French Open as well as a silver medalist in singles at the 2008 Beijing
Olympics.
</item 8>

<item 9>
Jelena Janković career statistics
This is a list of the main career statistics of Serbian professional tennis player and former World No. 1, Jelena Janković.

 To date, Janković has won 15 WTA singles titles including four WTA Tier I singles titles, one WTA Premier Mandatory singles title and one WTA Premier 5 singles
title.

 Janković was also the runner-up in singles at the 2008 US Open, a semi-finalist at the year-ending WTA Tour Championships in 2008, 2009 and 2013 and a
quarterfinalist at the 2008 Beijing Olympics.
</item 9>

<item 10>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 10>
=== END ITEM id=8d8ec917a7df25de ===

=== ITEM id=11bca91ee0a12b9e ===
Gold sentences and facts:
Sentence 0: Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
  Fact 0.0: Ruggero Deodato was born on 7 May 1939.
  Fact 0.1: Ruggero Deodato is an Italian film director.
  Fact 0.2: Ruggero Deodato is an Italian screenwriter.
  Fact 0.3: Ruggero Deodato is an Italian actor.
Sentence 1: Alejandro Springall is a Mexican film director and producer.
  Fact 1.0: Alejandro Springall is a Mexican film director.
  Fact 1.1: Alejandro Springall is a Mexican producer.

Retrieved items:
<item 1>
Alejandro Springall
Alejandro Springall is a Mexican film director and producer.
</item 1>

<item 2>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 2>

<item 3>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 3>

<item 4>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 4>

<item 5>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 5>

<item 6>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 6>

<item 7>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 7>

<item 8>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 8>

<item 9>
Cannibal Holocaust
Cannibal Holocaust is a 1980 Italian cannibal horror film directed by Ruggero Deodato from a screenplay by Gianfranco Clerici.

 It stars Carl Gabriel Yorke, Robert Kerman, Francesca Ciardi, Perry Pirkanen, and Luca Barbareschi.

 Influenced by the works of Mondo director Gualtiero Jacopetti, the film was inspired by Italian media reporting of Red Brigade terrorism.

 The coverage included news reports Deodato believed to be staged, an idea which became an integral aspect of the film's story.

 "Cannibal Holocaust" was filmed primarily in the Amazon rainforest of Colombia with indigenous tribes interacting with American and Italian actors.
</item 9>

<item 10>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 10>
=== END ITEM id=11bca91ee0a12b9e ===

=== ITEM id=5b620d3ad3bf5bb0 ===
Gold sentences and facts:
Sentence 0:  It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that position for three weeks.
  Fact 0.0: The song was released as the lead single from the fourth studio album of its act, "Elephant."
  Fact 0.1: The song was released in March 2003.
  Fact 0.2: The song reached number one on the Modern Rock Tracks.
  Fact 0.3: The song maintained its number one position on the Modern Rock Tracks for three weeks.
Sentence 1:  Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album charts.
  Fact 1.0: The album "Elephant" was released on April 1, 2003.
  Fact 1.1: The album "Elephant" was released on V2 Records.
  Fact 1.2: The album "Elephant" garnered near unanimous critical acclaim.
  Fact 1.3: The album "Elephant" garnered commercial success.
  Fact 1.4: The album "Elephant" received a nomination for Album of the Year at the 46th Grammy Awards in 2004.
  Fact 1.5: The album "Elephant" won Best Alternative Music Album at the 46th Grammy Awards in 2004.
  Fact 1.6: The album "Elephant" peaked at No. 6 on the US "Billboard" charts.
  Fact 1.7: The album "Elephant" topped the UK album charts.

Retrieved items:
<item 1>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 1>

<item 2>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 2>

<item 3>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 3>

<item 4>
Seven Nation Army
"Seven Nation Army" (also stylized as "7 Nation Army") is a song by American rock duo The White Stripes.

 It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that
position for three weeks.

 It also became the third best-performing song of the decade on the same chart.

 It was well received commercially as well, and won the Grammy Award for Best Rock Song.
</item 4>

<item 5>
The Boys Are Back (The Oak Ridge Boys album)
The Boys Are Back is the thirtieth studio album of country music group The Oak Ridge Boys.

 It was released in 2009 under the Spring Hill Music Group label.

 The album marked the group's return to secular country music after releasing gospel albums since 1992.

 The track "Seven Nation Army," a cover of The White Stripes's 2003 song from their album "Elephant", was released as the album's first single.
</item 5>

<item 6>
7 Seconds (film)
7 Seconds is a 2005 American crime film directed by Simon Fellows, starring Wesley Snipes and Tamzin Outhwaite.

 The film was released on Direct-to-DVD in the United States on June 28, 2005.

 The title refers to the timers at the beginning of the film, which are set at 00:07 (7 seconds).
</item 6>

<item 7>
Meg White
Megan Martha "Meg" White (born December 10, 1974) is an American drummer and occasional singer known for her work with Jack White in the Detroit rock duo The
White Stripes.

 On an impulse, she played on Jack's drums in 1997.

 The two decided to form a band and began performing two months later, calling themselves The White Stripes because of their last name and Meg's affinity for
peppermint candy.

 The band quickly became a Detroit underground favorite, before reaching national, then international fame.

 White has been nominated for various awards as a part of the White Stripes, and has received four Grammy Awards.
</item 7>

<item 8>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 8>

<item 9>
Late Night Tales: The Flaming Lips
Late Night Tales: The Flaming Lips is a compilation album compiled by the members of The Flaming Lips, featuring songs by various artists.

 The album was released on March 7, 2005 and it features one new Flaming Lips recording, a cover of the White Stripes' "Seven Nation Army".
</item 9>

<item 10>
Premio Lo Nuestro 2003
Premio Lo Nuestro 2003 was the 15th anniversary of the awards.

 the show was hosted by Mexican presenters Marco Antonio Regil and Adal Ramones.

 Juanes, Thalía, Marc Anthony, Pilar Montenegro, Sin Bandera, Banda el Recodo and other Latin music greats gave electrifying performances.

 In the show, there was 36 awards winners with 135 nominations.

 In Pop genre, Awards was given for : Album of the Year, Best Male Artist, Best Female Artist, Best Group or duo, Best New Artist and Song of the Year.

 In Rock Genre : Best Rock Album and Best rock Performer of the Year.

 In Tropical genre : Best Tropical Album of the Year, Best Tropical Male Artist, Best Tropical Female Artist, Best Tropical Group or Duo of the Year, Best
Tropical New Artist, Tropical Song of the Year, Best Merengue Performance, Best Salsa Performance and Best Traditional Performance.

 Juanes was the biggest winner of night, took home four awards Best Pop Male Artist, Best Music Video, Best Rock Performance, and Pop Song of the Year .
</item 10>
=== END ITEM id=5b620d3ad3bf5bb0 ===

=== ITEM id=23bb2e5673187852 ===
Gold sentences and facts:
Sentence 0: Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.
  Fact 0.0: Eric Arthur Blair was born on 25 June 1903.
  Fact 0.1: Eric Arthur Blair died on 21 January 1950.
  Fact 0.2: Eric Arthur Blair's pen name was George Orwell.
  Fact 0.3: Eric Arthur Blair is better known as George Orwell.
  Fact 0.4: Eric Arthur Blair was an English novelist.
  Fact 0.5: Eric Arthur Blair was an English essayist.
  Fact 0.6: Eric Arthur Blair was an English journalist.
  Fact 0.7: Eric Arthur Blair was an English critic.
Sentence 1: Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.
  Fact 1.0: Nelson Richard DeMille was born on August 23, 1943.
  Fact 1.1: Nelson Richard DeMille is an American author.
  Fact 1.2: Nelson Richard DeMille writes action adventure novels.
  Fact 1.3: Nelson Richard DeMille writes suspense novels.
Sentence 2:  DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
  Fact 2.0: Nelson Richard DeMille has written under the pen name Jack Cannon.
  Fact 2.1: Nelson Richard DeMille has written under the pen name Kurt Ladner.
  Fact 2.2: Nelson Richard DeMille has written under the pen name Ellen Kay.
  Fact 2.3: Nelson Richard DeMille has written under the pen name Brad Matthews.

Retrieved items:
<item 1>
The Panther (novel)
The Panther is a 2012 novel by American author Nelson DeMille.

 It is the sixth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "The Lion".

 The Panther is followed by DeMille’s 2015 novel, "Radiant Angel".

 Also featured in this novel is DeMille’s other fictional character, Paul Brenner, who appears in The General's Daughter and Up Country.
</item 1>

<item 2>
George Orwell: A Life in Pictures
George Orwell: A Life in Pictures is a 2003 BBC Television docudrama telling the life story of the British author George Orwell.

 Chris Langham plays the part of Orwell.

 No surviving sound recordings or video of the real George Orwell have been found.
</item 2>

<item 3>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 3>

<item 4>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 4>

<item 5>
The Lion's Game
The Lion's Game is a 2000 novel by American author Nelson DeMille.

 It is the second of DeMille's novels to feature the detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New
York City.

 The 2000 novel The Lion’s Game is the sequel to Plum Island and is followed by the 2004 book Night Fall.

 The book also briefly mentions events from other DeMille novels like "The Charm School" and "The Gold Coast", that aren't strictly part of the Corey-universe.
</item 5>

<item 6>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 6>

<item 7>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 7>

<item 8>
Wild Fire (novel)
Wild Fire is a 2006 novel by American author, Nelson DeMille.

 It is the fourth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to Night Fall and takes place approximately one year later.

 Wild Fire is followed by DeMille’s 2010 novel, The Lion.
</item 8>

<item 9>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 9>

<item 10>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 10>
=== END ITEM id=23bb2e5673187852 ===

=== ITEM id=c3685a59366adaf8 ===
Gold sentences and facts:
Sentence 0: The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.
  Fact 0.0: The Livesey Hall War Memorial commemorates the fallen of World War I.
  Fact 0.1: The Livesey Hall War Memorial commemorates the fallen of World War II.
  Fact 0.2: The commemorated fallen had been employed by the South Suburban Gas Company of London.
Sentence 1:  Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).
  Fact 1.0: Over 60 million people were killed.
  Fact 1.1: The deaths were about 3% of the 1940 world population.
  Fact 1.2: The estimated 1940 world population was 2.3 billion.

Retrieved items:
<item 1>
Livesey Hall War Memorial
The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.

 It is also a tribute to those employees who served in the wars.

 The monument was designed and executed by British sculptor Sydney March, of the March family of artists.
</item 1>

<item 2>
Vietnam War Memorial, Hanoi
The War Memorial in Hanoi is located across the Ba Dinh Square, across the Ho Chi Minh Mausoleum and close to Hanoi Citadel.

 Constructed in 1993 in a fusion of traditional Vietnamese and modernist architecture, the memorial commemorates men and women who sacrificed themselves during
the Second Indochina War.

 The war is known by many names, e.g. as the American War in Vietnam.

 The memorial is a focal point for state functions, commemorating the war dead.
</item 2>

<item 3>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 3>

<item 4>
Rainham War Memorial
The Rainham War Memorial commemorates soldiers killed in both World Wars as well as civilian casualties of World War II.
</item 4>

<item 5>
Alamein Memorial
The Alamein Memorial is a Commonwealth War Graves Commission war memorial in the El Alamein War Cemetery, El Alamein, Egypt.

 The memorial commemorates 11,866 Commonwealth forces members who died during World War II.

 The memorial was designed by Hubert Worthington and unveiled by Viscount Montgomery of Alamein on 24 October 1954.
</item 5>

<item 6>
World War II casualties
World War II was the deadliest military conflict in history in absolute terms of total casualties.

 Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).

 The tables below give a detailed country-by-country count of human losses.

 World War II fatality statistics vary, with estimates of total deaths ranging from 50 million to more than 80 million.

 The higher figure of over 80 million includes deaths from war-related disease and famine.

 Civilians killed totalled 50 to 55 million, including 19 to 28 million from war-related disease and famine.

 Military deaths from all causes totalled 21 to 25 million, including deaths in captivity of about 5 million prisoners of war.
</item 6>

<item 7>
Portsmouth Naval Memorial
The Portsmouth Naval Memorial, sometimes known as Southsea Naval Memorial, is a war memorial in Portsmouth, Hampshire, England, on Southsea Common beside
Clarence Esplanade, between Clarence Pier and Southsea Castle.

 The memorial commemorates approximately 25,000 British and Commonwealth sailors who were lost in the World Wars, around 10,000 sailors in the First World War
and 15,000 in the Second World War.

 The memorial features a central obelisk, with names of the dead on bronze plaques arranged around the memorial according to the year of death.
</item 7>

<item 8>
Tower Hill Memorial
The Tower Hill Memorial is a Commonwealth War Graves Commission war memorial on the south side of Trinity Square Gardens, in London, England.

 The memorial commemorates those from the Merchant Navy and fishing fleets who died during both world wars and have "no grave but the sea".

 The memorial was designed by Edwin Lutyens with sculpture work by William Reid Dick, the Second World War extension was designed by Edward Maufe with sculpture
work by Charles Wheeler.
</item 8>

<item 9>
Freetown Memorial
The Freetown Memorial is a Commonwealth War Graves Commission war memorial located outside the Secretariate Building in Freetown, Sierra Leone.

 The memorial generally commemorates Soldiers from Sierra Leone who died whilst serving with the Royal West African Frontier Force in West Africa and whose
graves are not known.

 The Freetown Memorial commemorates 1,109 First World War casualties and 247 from the Second World War.
</item 9>

<item 10>
Arras Flying Services Memorial
The Arras Flying Services Memorial Commonwealth War Graves Commission war memorial in the Faubourg d'Amiens Cemetery, Arras, France.

 The memorial commemorates nearly 1,000 airmen from forces of the Commonwealth who were killed on the Western Front during World War I and who have no known
grave.

 The memorial was designed by Edwin Lutyens, sculpted by William Reid Dick and unveiled by Hugh Trenchard, 1st Viscount Trenchard, Marshal of the Royal Air
Force on 31 July 1932.
</item 10>
=== END ITEM id=c3685a59366adaf8 ===

=== ITEM id=6141aa021f97724e ===
Gold sentences and facts:
Sentence 0:  This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's "Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
  Fact 0.0: This episode is a milestone in the history of the series.
  Fact 0.1: In this episode, Snowball II is killed off.
  Fact 0.2: Homer's mother died in season 19's Mona Leaves-a.
  Fact 0.3: Excluding the death of Homer's mother, the killing of Snowball II is to date the closest thing to an actual Simpson family member being killed off.
Sentence 1: Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".
  Fact 1.0: Mona Penelope Simpson's maiden name is Olsen.
  Fact 1.1: Mona Penelope Simpson is a recurring fictional character.
  Fact 1.2: Mona Penelope Simpson is a character in the animated television series The Simpsons.

Retrieved items:
<item 1>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 1>

<item 2>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 2>

<item 3>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 3>

<item 4>
My Mother the Carjacker
"My Mother the Carjacker" is the second episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It first aired on the Fox network in the United States on November 9, 2003.

 Homer receives a cryptic message in the newspaper informing him to come to a certain place at midnight, and soon discovers that the person who wrote the
message is his mother, Mona Simpson.

 It was written by Michael Price and directed by Nancy Kruse.

 Glenn Close makes her second of six guest spots as Homer's mother.

 It has a direct link from the season seven episode, "Mother Simpson".

 It was nominated for a Writers Guild of America Award in 2004.

 In its original run, the episode received 12.4 million viewers.
</item 4>

<item 5>
Girly Edition
"Girly Edition" is the twenty-first episode in the ninth season of the American animated television series "The Simpsons".

 It originally aired on the Fox network in the United States on April 19, 1998.

 In the episode, Lisa and Bart Simpson must co-anchor a new news program, though when Bart is seen as a more successful news anchor, Lisa becomes jealous and
seeks revenge.

 Meanwhile, in the subplot, Homer Simpson gets a monkey helper because of his laziness.

 "Girly Edition" was the first episode written by Larry Doyle and was directed by Mark Kirkland.

 Much of the subplot was inspired by the film "Monkey Shines".

 Critics gave the episode positive reviews and it was well received by Lisa's voice actress Yeardley Smith.
</item 5>

<item 6>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 6>

<item 7>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 7>

<item 8>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 8>

<item 9>
Lisa the Drama Queen
"Lisa the Drama Queen" is the ninth episode of the twentieth season of "The Simpsons".

 It originally aired on the Fox network in the United States on January 25, 2009 and guest starred Emily Blunt as Juliet.

 A special version of the end credits theme was performed by Fall Out Boy, although they do not guest star in the actual episode.

 This was the last episode in the series to be presented only in standard definition, the first regular episode to begin right after the opening credits without
a commercial break, with an episode now having four segments, and the last episode to use the original opening sequence starting from Season 2.

 Brian Kelley returns as a writer after five seasons of being absent from the show.

 The episode is very loosely based on the 1994 film "Heavenly Creatures".
</item 9>

<item 10>
I, (Annoyed Grunt)-bot
"I, (Annoyed Grunt)-Bot", also known as ""I, D'oh-Bot"", is the ninth episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It originally aired on the Fox network in the United States on January 11, 2004.

 This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's
"Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
</item 10>
=== END ITEM id=6141aa021f97724e ===

=== ITEM id=55fab257da1f90d5 ===
Gold sentences and facts:
Sentence 0: Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story writer.
  Fact 0.0: Doris May Lessing, CH, was born Doris May Tayler (née Tayler).
  Fact 0.1: Doris May Lessing was born on 22 October 1919.
  Fact 0.2: Doris May Lessing died on 17 November 2013.
  Fact 0.3: Doris May Lessing was a British novelist.
  Fact 0.4: Doris May Lessing was a British poet.
  Fact 0.5: Doris May Lessing was a British playwright.
  Fact 0.6: Doris May Lessing was a British librettist.
  Fact 0.7: Doris May Lessing was a British biographer.
  Fact 0.8: Doris May Lessing was a British short story writer.
Sentence 1: Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
  Fact 1.0: Philip Kindred Dick was born on December 16, 1928.
  Fact 1.1: Philip Kindred Dick died on March 2, 1982.
  Fact 1.2: Philip Kindred Dick was an American writer.
  Fact 1.3: Philip Kindred Dick is notable for publishing works of science fiction.

Retrieved items:
<item 1>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 1>

<item 2>
Doris Lessing
Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story
writer.

 Her novels include "The Grass Is Singing" (1950), the sequence of five novels collectively called "Children of Violence" (1952–69), "The Golden Notebook"
(1962), "The Good Terrorist" (1985), and five novels collectively known as "Canopus in Argos: Archives" (1979–1983).
</item 2>

<item 3>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 3>

<item 4>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 4>

<item 5>
Jenny Diski
Jenny Diski FRSL (née Simmonds; 8 July 1947 – 28 April 2016) was an English writer.

 She had a troubled childhood, but was rescued by the older novelist Doris Lessing; she lived in Lessing's house for four years.

 Diski was educated at University College London, and worked as a teacher during the 1970s and early 1980s.
</item 5>

<item 6>
The Father-Thing (collection)
The Father-Thing is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1989 and reprints Volume III of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Science Fiction Quarterly" and "Imaginative Tales".
</item 6>

<item 7>
Adjustment Team
"Adjustment Team" is a science fiction short story by American writer Philip K. Dick.

 It was first published in "Orbit Science Fiction" (September–October 1954, No. 4) with illustration by Faragasso.

 It was later reprinted in "The Sands of Mars and Other Stories" (Australian) in 1958, "The Book of Philip K. Dick" in 1973, "The Turning Wheel and Other
Stories" (United Kingdom) in 1977, "The Collected Stories of Philip K. Dick" in 1987 (Underwood–Miller), 1988 (Gollancz, United Kingdom), 1990 (Citadel
Twilight, United States), "Selected Stories of Philip K. Dick" in 2002 and in "The Early Work of Philip K. Dick, Volume One: The Variable Man & Other Stories"
in 2009.
</item 7>

<item 8>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 8>

<item 9>
The Days of Perky Pat (collection)
The Days of Perky Pat is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1990 and reprints Volume IV of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 The stories had originally appeared in the magazines "Galaxy Science Fiction", "Science Fiction Stories", "If", "Fantastic Universe", "Fantasy and Science
Fiction", "Fantastic", "Worlds of Tomorrow", "Escapade" and "Amazing Stories".
</item 9>

<item 10>
The Philip K. Dick Reader
The Philip K. Dick Reader is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Citadel Twilight in 1997.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Imaginative Tales", "Fantastic Universe" and "Space Science
Fiction".

 It is identical in content and order to the edition of volume 3 of the Collected Stories of Philip K. Dick produced by the same publisher apart from the
substitution of three stories in positions 21-23 of 24 and the omission of the end notes in the Collected Stories edition.

 At press time, stories 21 and 24 had already been made into successful movie adaptations and stories 22 and 23 had been optioned.
</item 10>
=== END ITEM id=55fab257da1f90d5 ===

=== ITEM id=f56adf533f17e35a ===
Gold sentences and facts:
Sentence 0: Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of Pinocchio" by Carlo Collodi.
  Fact 0.0: Pinocchio is a 1940 American film.
  Fact 0.1: Pinocchio is an animated musical fantasy film.
  Fact 0.2: Pinocchio was produced by Walt Disney Productions.
  Fact 0.3: Pinocchio is based on the Italian children's novel 'The Adventures of Pinocchio'.
  Fact 0.4: The novel 'The Adventures of Pinocchio' was written by Carlo Collodi.
Sentence 1:  It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
  Fact 1.0: Pinocchio was the second animated feature film produced by Disney.
  Fact 1.1: Pinocchio was made after the success of 'Snow White and the Seven Dwarfs'.
  Fact 1.2: Snow White and the Seven Dwarfs is from 1937.
Sentence 2: Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.
  Fact 2.0: Home on the Range is a 2004 American film.
  Fact 2.1: Home on the Range is an animated musical western comedy film.
  Fact 2.2: Home on the Range was produced by Walt Disney Feature Animation.
  Fact 2.3: Home on the Range was released by Walt Disney Pictures.

Retrieved items:
<item 1>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 1>

<item 2>
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
</item 2>

<item 3>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 3>

<item 4>
Graph query row:
first_work: Pinocchio
first_kind: film
first_published_on: 1940
second_work: Home on the Range
second_kind: film
second_published_on: 2004
chronological_order: Pinocchio before Home on the Range
connection_types: None
connected_entities: None
</item 4>

<item 5>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 5>

<item 6>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 6>

<item 7>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 7>

<item 8>
Home on the Range (2004 film)
Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.

 The 45th Disney animated feature film, it was the last 2D animated Disney film released until "The Princess and the Frog" in 2009.

 Named after the popular country song of the same name, "Home on the Range" features the voices of Roseanne Barr, Judi Dench, Jennifer Tilly, Cuba Gooding Jr.,
Randy Quaid, and Steve Buscemi.

 The film is set in the Old West, and centers on a mismatched trio of dairy cows—brash, adventurous Maggie; prim, proper Mrs. Caloway; and ditzy, happy-go-lucky
Grace.

 The three cows must capture an infamous cattle rustler named Alameda Slim for his bounty in order to save their idyllic farm from foreclosure.

 Aiding them in their quest is Lucky Jack, a feisty, peg-legged rabbit, but a selfish horse named Buck, eagerly working in the service of Rico, a famous bounty
hunter, seeks the glory for himself.
</item 8>

<item 9>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 9>

<item 10>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 10>
=== END ITEM id=f56adf533f17e35a ===
