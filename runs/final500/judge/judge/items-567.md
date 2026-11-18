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

=== ITEM id=723e2e079a24a855 ===
Gold sentences and facts:
Sentence 0: Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost figures of the French Avant-garde theatre.
  Fact 0.0: Eugène Ionesco was born Eugen Ionescu.
  Fact 0.1: Eugène Ionesco was born on 26 November 1909.
  Fact 0.2: Eugène Ionesco died on 28 March 1994.
  Fact 0.3: Eugène Ionesco was a Romanian-French playwright.
  Fact 0.4: Eugène Ionesco wrote mostly in French.
  Fact 0.5: Eugène Ionesco was one of the foremost figures of the French Avant-garde theatre.
Sentence 1: Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."
  Fact 1.0: Paul Mark Scott was born on 25 March 1920.
  Fact 1.1: Paul Mark Scott died on 1 March 1978.
  Fact 1.2: Paul Mark Scott was an English novelist.
  Fact 1.3: Paul Mark Scott was an English playwright.
  Fact 1.4: Paul Mark Scott was an English poet.
  Fact 1.5: Paul Mark Scott is best known for his monumental tetralogy "The Raj Quartet."

Retrieved items:
<item 1>
The New Tenant
The New Tenant (French: "Le Nouveau Locataire" ) is a play written by Eugène Ionesco in 1955.

 The central image is common to many Ionesco plays: something accumulates on stage and overwhelms the characters.

 In this case its furniture.

 The main characters are a gentleman, a caretaker, and two movers.

 The caretaker talks as the gentleman, the "new tenant" of the title, directs the two movers who continuously bring in furniture.
</item 1>

<item 2>
Paul Scott (novelist)
Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."

 His novel "Staying On" won the Booker Prize for 1977.
</item 2>

<item 3>
Exit the King
Exit the King (French: Le Roi se meurt ) is an absurdist drama by Eugène Ionesco that premiered in 1962.

 It is the third in Ionesco's "Berenger Cycle", preceded by "The Killer" (1958) and "Rhinocéros" (1959), and followed by "A Stroll in the Air" (1963).
</item 3>

<item 4>
Eugène Ionesco
Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost
figures of the French Avant-garde theatre.

 Beyond ridiculing the most banal situations, Ionesco's plays depict the solitude and insignificance of human existence in a tangible way.
</item 4>

<item 5>
The Killer (play)
The Killer (French: Tueur sans gages , sometimes translated "The Killer without Reason" or "The Killer without Cause") is a play written by Eugène Ionesco in
1958.

 It is the first of Ionesco's Berenger plays, the others being "Rhinocéros" (1959), "Exit the King" (1962), and "A Stroll in the Air" (1963).
</item 5>

<item 6>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 6>

<item 7>
Bill Scott (author)
William Neville "Bill" Scott OAM (4 October 1923 – 22 December 2005) was an Australian author, folklorist, songwriter, poet and a collector of bush ballads and
Australian folk history.

 He has published anthologies of Australian bush songs, including the best selling book "The Complete Book of Australian Folklore" published in 1976.

 He was awarded the Order of Australia in 1992 for his contributions to folklore, folk music and Australian literature.

 He was considered a living treasure and his anthologies of songs and his donated collections continue his legacy.
</item 7>

<item 8>
The Lesson
The Lesson (French: "La Leçon" ) is a one-act play by French-Romanian playwright Eugène Ionesco.

 It was first performed in 1951 in a production directed by Marcel Cuvelier (who also played the Professor).

 Since 1957 it has been in permanent showing at Paris' Théâtre de la Huchette, on an Ionesco double-bill with The Bald Soprano.

 The play is regarded as an important work in the "Theatre of the Absurd".
</item 8>

<item 9>
Victims of Duty
Victims of Duty (French: Victimes du Devoir ) is a one-act play written in 1953 by French-Romanian playwright Eugène Ionesco.

 An early work, it has not received the notoriety of his other works.

 The play is in the Theatre of the Absurd style, of which Ionesco was a pioneer.
</item 9>

<item 10>
Lonely Planet (play)
Lonely Planet is a two character play written by Steven Dietz.

 The play tells the story of Jody and Carl, two gay men who live in an unnamed American city.

 The play was written during the midst of the AIDS epidemic, which is the central focus of the story, though ultimately the play sends the message that one
should pay attention to the world around them and realize its problems rather than shun it.

 The play heavily references the Eugène Ionesco comedy "The Chairs".
</item 10>
=== END ITEM id=723e2e079a24a855 ===

=== ITEM id=c8823846d4495a68 ===
Gold sentences and facts:
Sentence 0: The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a unique numeric identifier, referred to as a "DUNS number" to a single business entity.
  Fact 0.0: The Data Universal Numbering System is abbreviated as DUNS or D-U-N-S.
  Fact 0.1: The Data Universal Numbering System is a proprietary system.
  Fact 0.2: The Data Universal Numbering System is developed by Dun & Bradstreet (D&B).
  Fact 0.3: The Data Universal Numbering System is regulated by Dun & Bradstreet (D&B).
  Fact 0.4: The Data Universal Numbering System assigns a unique numeric identifier to a single business entity.
  Fact 0.5: The unique numeric identifier is referred to as a DUNS number.
Sentence 1:  Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
  Fact 1.0: Dun & Bradstreet is often referred to as D&B.
  Fact 1.1: The company's database contains more than 265 million business records.
  Fact 1.2: The company's database contains business records worldwide.

Retrieved items:
<item 1>
Data Universal Numbering System
The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a
unique numeric identifier, referred to as a "DUNS number" to a single business entity.

 It was introduced in 1963 to support D&B's credit reporting practice.

 It is standard worldwide.

 DUNS users include the European Commission, the United Nations, and the United States government.

 More than 50 global industry and trade associations recognize, recommend, or require DUNS.

 The DUNS database contains over 250 million entries for businesses throughout the world.
</item 1>

<item 2>
Graph query row:
entity: Data Universal Numbering System
relationship: CREATED_BY
related: Dun & Bradstreet
</item 2>

<item 3>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
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
DFDS
DFDS is Northern Europe's largest shipping and logistics company.

 The company's name is an abbreviation of Det Forenede Dampskibs-Selskab (literally "The United Steamship Company").

 DFDS was founded in 1866, when C.F. Tietgen merged the three biggest Danish steamship companies of that day.
</item 5>

<item 6>
Universal Numbering System
The Universal Numbering System is a dental notation system for associating information to a specific tooth, and is commonly used in the United Kingdom.
</item 6>

<item 7>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 7>

<item 8>
The Denver Dry Goods Company
The Denver Dry Goods Company, also known as "The Denver", was established in Denver, Colorado in 1879 by Michael.

 J. McNamara and L.H. Flanders as M.J. McNamara & Company and later The McNamara Dry Goods Company.

 The beginnings of the company can be traced back to 1876 when they had both worked as clerks at another dry goods store in Denver.

 In 1877, McNamara left that store and formed a partnership with Edgar H. Drew.

 After two years, Drew left and L. H. Flanders came on as co-owner.

 In 1893, McNamara turned the store's ownership over to Dennis Sheedy and Charles Kountz and in 1894, the company was reorganized under the name, "Denver Dry
Goods Company".

 For a while it was claimed to be the largest department store west of Chicago.

 A description on one postcard from 1916 read: "The Largest Store in the Central West, 400 Feet long-Seven Acres Floor Area, 1,200 Employees, A $1,500,000
Stock, 15th to 16th on California Street Denver Colorado".
</item 8>

<item 9>
Dental notation
Dental professionals, in writing or speech, use several different dental notation systems for associating information with a specific tooth.

 The three most common systems are the ISO System, Universal Numbering System, and Palmer notation method.

 The ISO system is used worldwide, and the Universal is used widely in the United States.

 The ISO System can be easily adapted to computerized charting.
</item 9>

<item 10>
United Nations (band)
The idea of the side project can be traced back to the early 2000s, but it was not until 2008 that United Nations took shape.

 Geoff Rickly has said the band started over tequila with Daryl Palumbo.

 At the start of the band, all members except for Rickly were under contracts with other record labels and were not legally permitted to work in projects
released by another, making it difficult for the official lineup of the band to be well known.

 However some members from the band's original line-up have been revealed through interviews and social media posts, such as Daryl Palumbo, Jonah Bayer and
Lukas Previn.

 Since Rickly was the only member whose primary band was not under contract, his name is the only one that could legally appear in press materials.

 Press photos of the band released in 2008 depict four people wearing Ronald Reagan masks.
</item 10>
=== END ITEM id=c8823846d4495a68 ===

=== ITEM id=f0ca4c78c8352689 ===
Gold sentences and facts:
Sentence 0: The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.
  Fact 0.0: The Dundee Canal was an industrial canal.
  Fact 0.1: The Dundee Canal was in Clifton and Passaic.
  Fact 0.2: Clifton and Passaic are in Passaic County, New Jersey.
Sentence 1:  It was built between 1858 and 1861 and ran parallel to the Passaic River.
  Fact 1.0: The Dundee Canal was built between 1858 and 1861.
  Fact 1.1: The Dundee Canal ran parallel to the Passaic River.
Sentence 2:  It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.
  Fact 2.0: It runs parallel to the coast.
  Fact 2.1: It is separated from the Atlantic Ocean by barrier beaches.
  Fact 2.2: The barrier beaches include Palm Beach Island.
Sentence 3:  The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
  Fact 3.0: The lagoon is connected to the Atlantic Ocean.
  Fact 3.1: The lagoon is connected to the Atlantic Ocean by two inlets.
  Fact 3.2: The two inlets are permanent.
  Fact 3.3: The two inlets are man-made.

Retrieved items:
<item 1>
Erie Canal Lock 52 Complex
Erie Canal Lock 52 Complex is a national historic district located at Port Byron and Mentz in Cayuga County, New York.

 The district includes two contributing buildings (the Erie House and the blacksmith shop / mule barn); three contributing engineering structures (Erie Canal
Lock 52, culvert, and canal prism of the enlarged Erie Canal); and archaeological sites associated with the canal operations.

 Lock 52 was constructed 1849-1853 as part of the Enlarged Erie Canal program.

 It remained in operation until the rerouting of the canal under the New York State Barge Canal System in 1917.

 The Erie House was built in 1894 and is a two story frame structure that housed a saloon and hotel.
</item 1>

<item 2>
Power canal
A Power Canal refers to a canal used for hydraulic power generation, rather than for transport of watercraft.

 The power canal was a major factor in the Industrial revolution in New England in the 19th century.

 Most early power canals were mill races used mechanically to transfer power directly from falling water to machinery in mill buildings.

 Later, the hydraulic power generated electricity locally for the same mill factories.

 These power canals were often filled in as electricity (transported by power lines) replaced the need for local water power, and road transport needs or city
expansion needs reclaimed the land.

 Some hydraulic power canals were transformed into local electric generators, but most were closed.

 Remains of power canals can be seen in old mill towns and are often protected as historical structures today.
</item 2>

<item 3>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 3>

<item 4>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 4>

<item 5>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 5>

<item 6>
Caisson lock
A caisson lock was invented in the late 18th century as a solution to minimise the use of great volumes of water required to raise and lower canal boats through
large height differences.

 It was normal to only raise and lower boats through small height differences of a few feet when traversing undulating terrain.

 A solution was required when either large height differences were encountered or water was in short supply.

 The Caisson (or Caisoon) was thought to be one solution.

 The technology of the time was not capable of achieving this type of construction economically with current building materials.

 This design attempted was a type of canal lock in which a narrowboat is floated into a sealed watertight box and raised or lowered between two different canal
water levels.

 It was designed primarily as a water-saving measure, and also was an attempt to minimise construction costs compared with other engineering solutions of the
time.

 In use it was capable of replacing up to seven conventional locks.

 Other design benefits were speed of boat descent/ascent, and only a little loss of water when operating compared with a conventional boat lock.
</item 6>

<item 7>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 7>

<item 8>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 8>

<item 9>
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 9>

<item 10>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 10>
=== END ITEM id=f0ca4c78c8352689 ===

=== ITEM id=b3e8c0b1749a7220 ===
Gold sentences and facts:
Sentence 0: Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
  Fact 0.0: Summer Magic is a 1963 Walt Disney Productions film.
  Fact 0.1: Summer Magic stars Hayley Mills.
  Fact 0.2: Summer Magic stars Burl Ives.
  Fact 0.3: Summer Magic stars Dorothy McGuire.
  Fact 0.4: Summer Magic is a story about a Boston widow and her children.
  Fact 0.5: In Summer Magic, the Boston widow and her children take up residence in a small town in Maine.
Sentence 1: Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.
  Fact 1.0: Around the World in 80 Days is a 2004 American film.
  Fact 1.1: Around the World in 80 Days is an action-adventure comedy family film.
  Fact 1.2: Around the World in 80 Days is based on Jules Verne's novel of the same name.

Retrieved items:
<item 1>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 1>

<item 2>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 2>

<item 3>
Around the World (1956 song)
"Around the World" was the theme tune from the 1956 movie "Around the World in 80 Days" In the film, only an instrumental version of the song appeared, although
the vocal version has become by far the better known one.

 The song was written by Harold Adamson and Victor Young; Young died in 1956, several weeks after the film's release and he received the Academy Award for Best
Music, Scoring of a Dramatic or Comedy Picture posthumously.

 Young's orchestral version was a #13 hit on the Billboard charts in 1957.

 The recording by Bing Crosby was the B-side of the Victor Young version in 1957, on Festival SP45-1274 in Australia, and was a joint charting success.
</item 3>

<item 4>
Around the World in 80 Days (1956 film)
Around the World in 80 Days (sometimes spelled as Around the World in Eighty Days) is a 1956 American epic adventure-comedy film starring Cantinflas and David
Niven , produced by the Michael Todd Company and released by United Artists.
</item 4>

<item 5>
Summer Holiday (1963 film)
Summer Holiday is a British CinemaScope and Technicolor musical film featuring singer Cliff Richard.

 The film was directed by Peter Yates (his debut), produced by Kenneth Harper.

 The original screenplay was written by Peter Myers and Ronald Cass (who also wrote most of the song numbers and lyrics).

 The cast includes Lauri Peters, Melvyn Hayes, Teddy Green, Jeremy Bulloch, Una Stubbs, Pamela Hart, Jacqueline Daryl, Lionel Murton, Madge Ryan, David Kossoff,
Nicholas Phipps, Ron Moody and The Shadows.

 Herbert Ross choreographed the musical numbers.
</item 5>

<item 6>
Love Begins at 20
Love Begins at 20 is a 1936 American comedy film directed by Frank McDonald and written by Dalton Trumbo and Tom Reed, based on the 1929 play "Broken Dishes" by
Martin Flavin.

 The film stars Hugh Herbert, Patricia Ellis, Warren Hull, Hobart Cavanaugh, Dorothy Vaughan and Clarence Wilson.

 The film was released by Warner Bros. on August 22, 1936.
</item 6>

<item 7>
Around the World in 80 Days (1988 film)
Around the World in 80 Days is an Australian 48-minute direct-to-video animated film from Burbank Films Australia.

 It was originally released in 1988.

 The film is based on Jules Verne's classic French novel, "Around the World in 80 Days", first published in 1873, and was adapted by Leonard Lee.

 It was produced by Roz Phillips and featured original music by Simon Walker.

 The film imitated BRB Internacional's Spanish 1981 series, "La vuelta al mundo de Willy Fog", in its use of anthropomorphic animals in the human roles.

 The copyright in this film is now owned by Pulse Distribution and Entertainment and administered by digital rights management firm NuTech Digital.
</item 7>

<item 8>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 8>

<item 9>
Summer Magic
Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children
taking up residence in a small town in Maine.

 The film was based on the novel "Mother Carey's Chickens" by Kate Douglas Wiggin and was directed by James Neilson.

 This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.
</item 9>

<item 10>
Around the World in 80 Days (2004 film)
Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.

 It stars Jackie Chan, Steve Coogan and Cécile de France.

 The film is set in 19th-century Britain and centers on Phileas Fogg (Steve Coogan), here reimagined as an eccentric inventor, and his efforts to circumnavigate
the globe in 80 days.

 During the trip, he is accompanied by his Chinese valet, Passepartout (Jackie Chan).

 For comedic reasons, the film intentionally deviated wildly from the novel and included a number of anachronistic elements.

 With production costs of about $110 million and estimated marketing costs of $30 million, it earned $24 million at the U.S. box office and $72 million
worldwide, making it a box office flop.

 It was Arnold Schwarzenegger's last film before he took a hiatus from acting to become Governor of California until 2010's "The Expendables".
</item 10>
=== END ITEM id=b3e8c0b1749a7220 ===

=== ITEM id=05f8908bb457b597 ===
Gold sentences and facts:
Sentence 0: Chester Erskine (November 29, 1905 – April 7, 1986) was a Hollywood and Broadway director, writer, and producer.
  Fact 0.0: Chester Erskine was born on November 29, 1905.
  Fact 0.1: Chester Erskine died on April 7, 1986.
  Fact 0.2: Chester Erskine was a Hollywood and Broadway director.
  Fact 0.3: Chester Erskine was a Hollywood and Broadway writer.
  Fact 0.4: Chester Erskine was a Hollywood and Broadway producer.
Sentence 1: Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.
  Fact 1.0: Sam Taylor was born on August 13, 1895, in New York City.
  Fact 1.1: Sam Taylor died on March 6, 1958, in Santa Monica.
  Fact 1.2: Sam Taylor was a film director.
  Fact 1.3: Sam Taylor was a screenwriter.
  Fact 1.4: Sam Taylor was a film producer.
  Fact 1.5: Sam Taylor was most active in the silent film era.

Retrieved items:
<item 1>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 1>

<item 2>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 2>

<item 3>
Speedy West
Wesley Webb West (January 25, 1924 – November 15, 2003), better known as Speedy West, was an American pedal steel guitarist and record producer.

 He frequently played with Jimmy Bryant, both in their own duo and as part of the regular Capitol Records backing band for Tennessee Ernie Ford and many others.

 He also played on Loretta Lynn's first single.
</item 3>

<item 4>
Chester Erskine
Chester Erskine (November 29, 1905 – April 7, 1986) was a Hollywood and Broadway director, writer, and producer.
</item 4>

<item 5>
William Farrar Smith
William Farrar Smith (February 17, 1824 – February 28, 1903), known as ‘Baldy’ Smith, was a Union general in the American Civil War, notable for attracting the
extremes of glory and blame.

 He was praised for his gallantry in the Seven Days Battles and the Battle of Antietam, but was demoted for insubordination after the disastrous defeat at the
Battle of Fredericksburg.

 As chief engineer of the Army of the Cumberland, he achieved recognition by restoring a supply line that saved that army from starvation and surrender, known
the “Cracker Line”, that helped Union troops to success in the Chattanooga Campaign in the autumn of 1863.

 Leading the first operation against Petersburg, Smith’s hesitation, possibly illness-related, cost the Union a prime opportunity for a quick end to the war,
and he was relieved of command.
</item 5>

<item 6>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 6>

<item 7>
Buford Abner
James Buford Abner (November 10, 1917 - November 19, 2011) was an American songwriter, musician and singer who worked during the early days of country music,
working in both secular and gospel country music genres.

 With his brother Merle Abner, his uncle Stacy Abner, George Hughes and Billy Carrier, he was a member of the Swanee River Boys.

 He was inducted into the Southern Gospel Music Association Hall of Fame in 2002 and the Atlanta Country Music Hall of Fame as a member of the Swanee River
Boys.
</item 7>

<item 8>
Dick Herrick
The Very Rev Richard William Herrick (3 December 1913 - 5 May 1981) was an eminent Anglican priest in the 20th century.

 He was educated at King Edward VI School Retford and Leeds University and was initially a civil servant.

 He was ordained after a period of study at the College of the Resurrection, Mirfield in 1940.

 he held curacies at Duston and Portsea, Portsmouth before being appointed Vicar of St Michael’s, Northampton in 1947, a post he held for a decade.

 He was then a Canon Residentiary of Chelmsford Cathedral until 1978 when he was appointed Provost of Chelmsford.

 He died in post.
</item 8>

<item 9>
Albert Sharpe
Albert Sharpe (15 April 1885 – 13 February 1970) was an Irish stage and film actor.

 His most famous roles were those of Darby O'Gill in Disney's "Darby O'Gill and the Little People" and as Finian McLonergan in the Original Broadway production
of the musical "Finian's Rainbow".

 (the film version, made in 1968, had Fred Astaire in the role.)

 On screen he played Fiona's father Andrew in the MGM musical Brigadoon.

 He was also a member of the Abbey Players.

 His last ten years were spent in retirement.

 He died in 1970 in Belfast at the age of 84.
</item 9>

<item 10>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 10>
=== END ITEM id=05f8908bb457b597 ===

=== ITEM id=6a77a96b22f28ac4 ===
Gold sentences and facts:
Sentence 0:  They have produced two studio albums and an EP.
  Fact 0.0: They have produced two studio albums.
  Fact 0.1: They have produced an EP.
Sentence 1: Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.
  Fact 1.0: Motion City Soundtrack was an American rock band.
  Fact 1.1: Motion City Soundtrack formed in Minneapolis, Minnesota.
  Fact 1.2: Motion City Soundtrack formed in 1997.
Sentence 2:  Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.
  Fact 2.0: Motion City Soundtrack had a nearly twenty-year career.
  Fact 2.1: Motion City Soundtrack toured heavily over the course of its career.
  Fact 2.2: Motion City Soundtrack released six studio albums.
  Fact 2.3: The majority of Motion City Soundtrack's six studio albums were released on the independent label Epitaph Records.

Retrieved items:
<item 1>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 1>

<item 2>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 2>

<item 3>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 3>

<item 4>
My Dinosaur Life
My Dinosaur Life is the fourth studio album by the American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on January 19, 2010 by Columbia Records.

 After many years on independent label Epitaph Records, Motion City Soundtrack signed to major label Columbia in 2006, prior to the release of their third
album, "Even If It Kills Me" (2007).

 Blink-182 bassist Mark Hoppus returned to produce the album; he had previously worked with the band on their 2005 breakthrough "Commit This to Memory".
</item 4>

<item 5>
Even If It Kills Me
Even if It Kills Me is the third studio album by American rock band Motion City Soundtrack.

 Produced by Ric Ocasek, Adam Schlesinger and Eli Janney, the album was released on September 18, 2007, in the United States by Epitaph Records.

 Motion City Soundtrack, based in Minneapolis, Minnesota, made a breakthrough with their second album, "Commit This to Memory", garnering praise and independent
buzz upon its 2005 release.

 Following the release, the band toured relentlessly over the next two years, during which time frontman Justin Pierre struggled with alcohol and substance
abuse.

 These addictions were infused into the writing process of "Even If Kills Me", which was recorded in early 2007 Stratosphere Studios in Chelsea and the
legendary Electric Lady Studios in Greenwich Village.
</item 5>

<item 6>
Commit This to Memory
Commit This to Memory is the second studio album by American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on June 7, 2005, in the United States by Epitaph Records.

 Motion City Soundtrack, formed in 1997, had first found success with their debut album, "I Am the Movie" (2002).

 The band toured in the interim years, creating positive word-of-mouth.

 In 2004, the band joined Blink-182 on the road for a string of shows, which led to their bassist, Mark Hoppus, joining the band in the studio for his first
producing effort.
</item 6>

<item 7>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 7>

<item 8>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 8>

<item 9>
Motion City Soundtrack discography
The following is the discography of Motion City Soundtrack, an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 9>

<item 10>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 10>
=== END ITEM id=6a77a96b22f28ac4 ===

=== ITEM id=cc665a918489ccb0 ===
Gold sentences and facts:
Sentence 0: Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.
  Fact 0.0: Catfish and the Bottlemen are a British rock band.
  Fact 0.1: Catfish and the Bottlemen were formed in Llandudno, Wales.
  Fact 0.2: Catfish and the Bottlemen were formed in 2007.
Sentence 1: The Lumineers are an American folk rock/Americana band based in Denver, Colorado.
  Fact 1.0: The Lumineers are an American folk rock/Americana band.
  Fact 1.1: The Lumineers are based in Denver, Colorado.
Sentence 2:  Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.
  Fact 2.0: Schultz and Fraites began writing together in 2005.
  Fact 2.1: Schultz and Fraites began performing together in 2005.
  Fact 2.2: Schultz and Fraites began writing together in Ramsey, New Jersey.
  Fact 2.3: Schultz and Fraites began performing together in Ramsey, New Jersey.
Sentence 3:  The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.
  Fact 3.0: The Lumineers emerged as one of the most popular folk rock/Americana artists.
  Fact 3.1: The Lumineers emerged during the revival of folk rock and Americana genres.
  Fact 3.2: The Lumineers emerged during the growing popularity of folk rock and Americana in the 2010s.

Retrieved items:
<item 1>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 1>

<item 2>
The Balcony (album)
The Balcony is the debut studio album by Welsh rock band Catfish and the Bottlemen.

 It was released on 15 September 2014 in the United Kingdom and 15 January 2015 in the United States.

 On 20 March 2015, "The Balcony" was awarded a Gold certification from the British Phonographic Industry.

 The album cover depicts outlines of two headless people pleasuring each other's genitals.

 The illustration is by New York artist Tim Lahan who originally posted it to his Flickr account in 2009 and was subsequently contacted by the band.
</item 2>

<item 3>
Billy Bibby &amp; The Wry Smiles
Billy Bibby & The Wry Smiles are a British indie rock band formed in the autumn of 2015 in the area around Llandudno, North Wales and Chester, UK.

 The band is fronted by Billy Bibby, former lead guitarist and founding member of Catfish and the Bottlemen.

 Bibby parted ways with Catfish and the Bottlemen in the summer of 2014, shortly after co-writing and recording their Platinum selling debut album, "The
Balcony".

 Soon thereafter, Billy Bibby began writing songs and playing acoustic shows in various UK cities in order to test public response to his new music.

 In the spring of 2015, Billy Bibby began seeking musicians to form a new band.

 Auditions were conducted with the help of Simon Jones, formerly of The Verve.

 Band members Rob Jones (lead guitar), Matt Bassy (bass), and Michael Pearce (drums) were selected to form the four-piece ensemble.
</item 3>

<item 4>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 4>

<item 5>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 5>

<item 6>
Kathleen (song)
"Kathleen" is the third single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was included in their EP, "Kathleen and the Other Three", and their debut studio album, "The Balcony".

 The single was digitally released on 18 March 2014 as a music video, and on 7 April 2014, the 7-inch version of the single was released.

 The single did not contain a B-side.
</item 6>

<item 7>
The Ride Tour
The Ride Tour is the ongoing second concert tour by English indie rock band Catfish and the Bottlemen, in support of the group's second studio album "The Ride".

 The tour was announced in February 2016 and began on 4 April 2016, in Glasgow, United Kingdom at the O2 Academy Glasgow.

 The tour is planned to travel across Europe and North America with a total of 50 shows planned so far.

 On 22 February 2016, the band announced an April 2016 stint of shows in UK, including Glasgow, Halifax, Southend On Sea, Doncaster, London and Brighton.

 The band will also headline several festivals across the UK throughout the summer, including Liverpool's 'Sound City' in May 2016, Bristol's Summer Series in
July 2016 and Manchester's Castlebowl in July 2016.

 On 14 March 2016 it was announced that the band will play several dates in North America in June 2016.
</item 7>

<item 8>
The Ride (Catfish and the Bottlemen album)
The Ride is the second studio album by British indie rock band Catfish and the Bottlemen.

 It was released on 27 May 2016 through Capitol Records.
</item 8>

<item 9>
Beautiful Decay
Beautiful Decay is the second extended play by Welsh indie rock band, Catfish and the Bottlemen.

 Released on 10 March 2010, the EP contained one song that would later be featured in their debut studio album, "The Balcony".
</item 9>

<item 10>
Soundcheck (song)
"Soundcheck" is the eighth single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was the lead single off of their sophomore album, "The Ride".

 The single was a promotional single that was digitally released on 16 February 2016.
</item 10>
=== END ITEM id=cc665a918489ccb0 ===

=== ITEM id=1c30905cc6a64a07 ===
Gold sentences and facts:
Sentence 0: Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).
  Fact 0.0: Dictyosperma is a monotypic genus of flowering plant.
  Fact 0.1: Dictyosperma is in the palm family.
  Fact 0.2: Dictyosperma is found in the Mascarene Islands in the Indian Ocean.
  Fact 0.3: Dictyosperma is found in Mauritius.
  Fact 0.4: Dictyosperma is found in Réunion.
  Fact 0.5: Dictyosperma is found in Rodrigues.
Sentence 1: The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in 1810.
  Fact 1.0: The genus Huernia is in the family Apocynaceae.
  Fact 1.1: The genus Huernia is in the subfamily Asclepiadoideae.
  Fact 1.2: The genus Huernia consists of stem succulents.
  Fact 1.3: The stem succulents of the genus Huernia are from Eastern Africa.
  Fact 1.4: The stem succulents of the genus Huernia are from Southern Africa.
  Fact 1.5: The genus Huernia was first described as a genus in 1810.

Retrieved items:
<item 1>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 1>

<item 2>
The entries describe diverse plants and one crab genus. Mascarene Dictyosperma album, the wind-shedding “hurricane palm,” is widely cultivated but nearly
extinct in its native range. Canary Islands Adenocarpus viscosus grows above 1,800 m. Dregea vines occur in Africa and southern Asia; Duranta shrubs and trees
are ornamental, with D. erecta invasive in several regions. Tropical Dieffenbachia are popular houseplants
</item 2>

<item 3>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 3>

<item 4>
Huernia is an African stem-succulent genus whose carrion-scented flowers attract flies. The Asian emerald ash borer devastates ash trees abroad. Oenanthe
javanica is an Asian water vegetable. California botanist Samuel Parish is commemorated by many plant names. Ardisia, Myrsine and Rapanea are Primulaceae
genera; Broughtonia is Caribbean orchids; Stephanocereus Brazilian cacti; Bognera a rare
</item 4>

<item 5>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 5>

<item 6>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 6>

<item 7>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 7>

<item 8>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 8>

<item 9>
Harrimanella
Harrimanella is a genus of flowering plant in the heath family Ericaceae, with a single species, Harrimanella hypnoides, also known as moss bell heather.

 It was originally named Cassiope hypnoides by Linnaeus (1737) in his Flora Lapponica, but "Harrimanella hypnoides" is now the accepted name at ITIS.

 The species name "hypnoides" means 'like "Hypnum" ', which is a genus mosses.
</item 9>

<item 10>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 10>
=== END ITEM id=1c30905cc6a64a07 ===

=== ITEM id=885250237b79e275 ===
Gold sentences and facts:
Sentence 0: The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.
  Fact 0.0: The Bishop of Ely is the ordinary of the Diocese of Ely.
  Fact 0.1: The Diocese of Ely is a Church of England diocese.
  Fact 0.2: The Diocese of Ely is in the Province of Canterbury.
Sentence 1:  Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
  Fact 1.0: Conway became Bishop of Ely in 2010.
  Fact 1.1: Conway was translated to Bishop of Ely from the Diocese of Salisbury.
  Fact 1.2: Conway was Bishop suffragan of Ramsbury before becoming Bishop of Ely.
  Fact 1.3: Conway was Bishop suffragan of Ramsbury in the Diocese of Salisbury.
Sentence 2: The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.
  Fact 2.0: The Diocese of Salisbury is a Church of England diocese.
  Fact 2.1: The Diocese of Salisbury is in the south of England.
  Fact 2.2: The Diocese of Salisbury is within the ecclesiastical Province of Canterbury.

Retrieved items:
<item 1>
Ecclesiastical Province of Ontario
The Ecclesiastical Province of Ontario is one of the Anglican Church of Canada's four ecclesiastical provinces.

 It was established in 1912 out of six dioceses of the Ecclesiastical Province of Canada located in the civil Province of Ontario, and the Diocese of Moosonee
from the Ecclesiastical Province of Rupert's Land.
</item 1>

<item 2>
Paul Burbridge
John Paul Burbridge was the Dean of Norwich in the latter part of the 20th century (1983-1995).

 Born on 21 May 1932, he was educated at The King's School, Canterbury and King's College, Cambridge, New College, Oxford, and Wells Theological College.

 After National Service with the Royal Artillery, he was ordained to a curacy at Eastbourne Parish Church in 1959.

 In 1962 he was appointed vicar choral and chamberlain at York Minster.

 He was appointed residentiary canon precentor at York Minster in 1966.

 In 1976 he was appointed Archdeacon of Richmond and canon residentiary at Ripon Cathedral, a post he held until his appointment as Dean of Norwich in 1983
(F.S.A).
</item 2>

<item 3>
Bishop of Ely
The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.

 The diocese roughly covers the county of Cambridgeshire (with the exception of the Soke of Peterborough), together with a section of north-west Norfolk and has
its episcopal see in the City of Ely, Cambridgeshire, where the seat is located at the Cathedral Church of the Holy Trinity.

 The current bishop is Stephen Conway, who signs "+Stephen Elien:" (abbreviation of the Latin adjective "Eliensis", meaning "of Ely").

 The diocesan bishops now reside in Bishop's House, Ely, the former cathedral deanery.

 Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
</item 3>

<item 4>
Roman Catholic Ecclesiastical Province of Miami
The Ecclesiastical Province of Miami is a Catholic ecclesiastical province covering the U.S. state of Florida.

 Its metropolitan bishop is the Archbishop of Miami, head of the Archdiocese of Miami.

 The province additionally includes the suffragan dioceses of Orlando, Palm Beach, Pensacola-Tallahassee, St. Augustine, St. Petersburg, and Venice.
</item 4>

<item 5>
Ecclesiastical Province of Freiburg
The Ecclesiastical Province of Freiburg ("Kirchenprovinz Freiburg") or Upper Rhenish Ecclesiastical Province ("Oberrheinische Kirchenprovinz") is an
ecclesiastical province of the Roman Catholic Church in the Upper Rhine area of Germany, centring on Freiburg im Breisgau.

 It covers the Roman Catholic Archdiocese of Freiburg, the Roman Catholic Diocese of Mainz and the Roman Catholic Diocese of Rottenburg-Stuttgart, covering
large areas of Baden-Württemberg and Hesse and small parts of Rhineland-Palatinate.

 Its metropolitan bishop is the Archbishop of Freiburg - that Archdiocese and the Province were both set up in 1821 in the wake of the 1801 Concordat and the
1815 Congress of Vienna.
</item 5>

<item 6>
Roman Catholic Archdiocese of Cashel and Emly
The Archdiocese of Cashel and Emly (Irish: "Ard-Deoise Chaisil agus Imligh" ) is a Roman Catholic archdiocese in mid-western Ireland.

 The archdiocese is led by the Archbishop of Cashel and Emly, who serves as pastor of the mother church, the Cathedral of the Assumption and metropolitan of the
ecclesiastical province of Cashel and Emly.

 The Diocese of Cashel was established in 1111 by the Synod of Rathbreasail.

 The ecclesiastical province, which was roughly co-extensive with the secular province of Munster, was created in 1152 by the Synod of Kells.

 The cathedral church of the archdiocese is located in Thurles, County Tipperary.

 The incumbent archbishop is Kieran O'Reilly.
</item 6>

<item 7>
An ecclesiastical province groups dioceses under a metropolitan archbishop or bishop. Examples include Bamenda, Cameroon (created in 1982; suffragans Buea,
Kumbo, and Mamfe; 35 parishes in six deaneries as of 2013); Freiburg, Germany (Freiburg, Mainz, and Rottenburg-Stuttgart); Miami, Florida (six suffragan
dioceses); and Owerri, Nigeria, which includes
</item 7>

<item 8>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 8>

<item 9>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 9>

<item 10>
Ecclesiastical province
An ecclesiastical province is a general term for one of the basic forms of jurisdiction in Christian Churches with traditional hierarchical structure, including
Western Christianity and Eastern Christianity.

 In general, ecclesiastical province is consisted of several dioceses (or eparchies), one of them being the archdiocese (or archeparchy), headed by metropolitan
bishop or archbishop who has ecclesiastical jurisdiction over all other bishops of the province.
</item 10>
=== END ITEM id=885250237b79e275 ===

=== ITEM id=a3052705fe1f6349 ===
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
Never Let Me Go (Florence and the Machine song)
"Never Let Me Go" is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by Florence Welch and Paul Epworth while production was handled by Epworth.

 Island Records released the song as the third single from the album on 30 March 2012.

 The band first premiered the song during one of their concerts before the release of the album on 15 October 2011.

 A Clams Casino remix of the song was placed on the 12" white vinyl single which was available for pre-order on 22 March 2012.
</item 1>

<item 2>
Lover to Lover
"Lover to Lover" is a song by English indie rock band Florence and the Machine.

 The song was released on 30 November 2012, as a digital download on iTunes as the fifth and final single from their second studio album "Ceremonials" (2011).

 The song was written by Florence Welch and Francis "Eg" White and produced by Paul Epworth.

 The Ceremonials Tour version of "Lover to Lover" is an edited version of the original with further production by Chris Hayden.

 The song became the band's first single to miss the top 200 in the UK Singles Chart.
</item 2>

<item 3>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 3>

<item 4>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 4>

<item 5>
What the Water Gave Me (painting)
What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.

 It is sometimes referred to as "What I Saw in the Water".
</item 5>

<item 6>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 6>

<item 7>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 7>

<item 8>
Wish That You Were Here
"Wish That You Were Here" is a song by the English indie rock band Florence and the Machine, written by Florence Welch, Andrew Wyatt, and Emilie Haynie, and was
released on Island Records on 26 August 2016.

 The song was made available via digital download, and is featured on the soundtrack of the film "Miss Peregrine's Home for Peculiar Children" which was
directed by Tim Burton.

 Welch — a long-time fan of Burton's work — had expressed interest in collaborating with him long before the recording of "Wish That You Were Here" as they
shared similar artistic themes.

 Upon release, the composition was positively received by music critics and reached number 128 on the UK Singles Chart.
</item 8>

<item 9>
Spectrum (Say My Name)
"Spectrum" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 It was written by lead singer Florence Welch and Paul Epworth with production being handled by Epworth.

 The band premiered the song during a concert at Brooklyn's Creators Project on 15 October 2011, prior to the release of the album.

 The album version of "Spectrum" is a downtempo orchestral pop, R&B, krautrock and disco song.

 The accompanying music video for the song premiered on 30 May 2012.

 It was directed by David LaChapelle and John Byrne.
</item 9>

<item 10>
Shake It Out
"Shake It Out" is a song by English indie rock band Florence and the Machine, released as the first official single from their second studio album,
"Ceremonials" (2011).

 It was written by Florence Welch and Paul Epworth, while production was handled by Epworth.

 The song was digitally released in Australia on 14 September 2011, and it was available in the United States on 19 October.

 It had its radio debut on XFM on 14 September 2011 in the United Kingdom.

 Welch revealed that the song was written within an hour and according to her it talked about shaking the regrets and the things that were haunting her.
</item 10>
=== END ITEM id=a3052705fe1f6349 ===

=== ITEM id=8ecf4f1c5cc98b2d ===
Gold sentences and facts:
Sentence 0: "Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".
  Fact 0.0: 'Into Dust' is a song by the American duo Mazzy Star.
  Fact 0.1: 'Into Dust' appears as the ninth track on the second album of Mazzy Star.
  Fact 0.2: The second album of Mazzy Star is 'So Tonight That I Might See'.
Sentence 1: Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and bassist Kendra Smith.
  Fact 1.0: Mazzy Star is an American alternative rock band.
  Fact 1.1: Mazzy Star was formed in Santa Monica, California, in 1989.
  Fact 1.2: Mazzy Star was formed from the group Opal.
  Fact 1.3: Opal was a collaboration of guitarist David Roback and bassist Kendra Smith.

Retrieved items:
<item 1>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 1>

<item 2>
Berlin (band)
Berlin is an American new wave band.

 The group was formed in Orange County in 1979 by John Crawford (bass guitar).

 Band members included Crawford, Terri Nunn (vocals), David Diamond (keyboards), Ric Olsen (guitar), Matt Reid (keyboards) and Rod Learned (drums).

 The band gained mainstream-commercial success in the early 1980s with singles including "The Metro", "Sex (I'm A..

.)

", "No More Words" and then in the mid 1980s with the chart-topping single "Take My Breath Away" from the 1986 film "Top Gun".
</item 2>

<item 3>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 3>

<item 4>
Profiles cover rock and metal acts, musicians, albums, and songs. Candlelight Red’s “Demons” and *Reclamation* were produced by Sevendust’s Morgan Rose; Louis
XIV reunited after a 2009 breakup. Mazzy Star’s sparse “Into Dust” features acoustic guitar and violin. Other highlights include Assembly of Dust’s Reid
Genauer, American Hi-Fi’s pop-punk mix, Mastodon’s *Leviathan*, Pantera’s shift to groove metal, and
</item 4>

<item 5>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 5>

<item 6>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 6>

<item 7>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 7>

<item 8>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 8>

<item 9>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 9>

<item 10>
Into Dust
"Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".

 The song's arrangement is sparse and features only acoustic guitar and violin as accompaniment to the vocals.
</item 10>
=== END ITEM id=8ecf4f1c5cc98b2d ===
