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

=== ITEM id=a989891f9f13748e ===
Gold sentences and facts:
Sentence 0: Basie Meets Bond is a 1966 album by Count Basie and his orchestra.
  Fact 0.0: Basie Meets Bond is a 1966 album.
  Fact 0.1: Basie Meets Bond is an album by Count Basie.
  Fact 0.2: Basie Meets Bond is an album by Count Basie and his orchestra.
Sentence 1:  The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".
  Fact 1.0: The album is a collection of musical pieces.
  Fact 1.1: The album's musical pieces come from the first four James Bond films.
  Fact 1.2: The first four James Bond films include Dr No.
  Fact 1.3: The first four James Bond films include From Russia with Love.
  Fact 1.4: The first four James Bond films include Goldfinger.
  Fact 1.5: The first four James Bond films include Thunderball.
Sentence 2: William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.
  Fact 2.0: William James 'Count' Basie lived from August 21, 1904 to April 26, 1984.
  Fact 2.1: Count Basie was an American jazz pianist.
  Fact 2.2: Count Basie was an American organist.
  Fact 2.3: Count Basie was an American bandleader.
  Fact 2.4: Count Basie was an American composer.

Retrieved items:
<item 1>
Count Basie/Sarah Vaughan
Count Basie/Sarah Vaughan is a 1961 album by American jazz singer Sarah Vaughan, accompanied by the Count Basie Orchestra, with arrangements by Frank Foster,
Thad Jones and Ernie Wilkins.

 According to James Gavin's liner notes to the 1996 CD release, Basie himself does not perform on any of the tracks.
</item 1>

<item 2>
Basie's Beatle Bag
Basie's Beatle Bag is a 1966 studio album by Count Basie and his orchestra, arranged by Chico O'Farrill.

 Basie released a second album of Beatles songs, "Basie on the Beatles", in 1969.
</item 2>

<item 3>
Basie Meets Bond
Basie Meets Bond is a 1966 album by Count Basie and his orchestra.

 The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".

 The album was Basie's first for United Records, and was produced by Teddy Reig.
</item 3>

<item 4>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 4>

<item 5>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 5>

<item 6>
Count Basie at Newport
Count Basie at Newport is a live album by jazz musician Count Basie and his orchestra.

 It was originally issued as Verve MGV 8243 and included only the tracks 1-8 and 13.

 Tracks 9-12 originally included in "Count Basie & Joe Williams/Dizzy Gillespie & Mary Lou Williams at Newport" (Verve MGV 8244).
</item 6>

<item 7>
Basie (album)
Basie (reissued as The Band of Distinction) is an album by pianist/bandleader Count Basie recorded in 1954 and originally released on the Clef label.

 the album should not be confused with Basie's 1958 album which became known as "The Atomic Mr. Basie".
</item 7>

<item 8>
Count Basie
William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.

 His mother taught him to play the piano and he started performing in his teens.

 Dropping out of school, he learned to operate lights for vaudeville and to improvise accompaniment for silent films at a local movie theater in his home town
of Red Bank, New Jersey.

 By age 16, he increasingly played jazz piano at parties, resorts and other venues.

 In 1924, he went to Harlem, where his performing career expanded; he toured with groups to the major jazz cities of Chicago, St. Louis and Kansas City.

 In 1929 he joined Bennie Moten's band in Kansas City, and played with them until Moten's death in 1935.
</item 8>

<item 9>
The Atomic Mr. Basie
The Atomic Mr. Basie (originally called Basie, also known as E=MC and reissued in 1994 as The Complete Atomic Basie) is a 1958 album by Count Basie and his
orchestra.

 The album is one of Basie's most famous and is critically acclaimed.

 Allmusic gave it 5 stars, reviewer Bruce Eder saying "it took Basie's core audience and a lot of other people by surprise, as a bold, forward-looking statement
within the context of a big-band recording."

 It is included in the book "1001 Albums You Must Hear Before You Die", Will Fulford-Jones calling it "Basie's last great record."

 According to the 2015 site update of Acclaimed Music, it is the 6th most critically acclaimed album of 1958, the 25th most acclaimed of the 1950s, and the
837th most acclaimed of all time, based on an aggregation of hundreds of critics' lists from around the world.
</item 9>

<item 10>
Basie on the Beatles
Basie on the Beatles is an album by pianist and bandleader Count Basie featuring performances recorded in late 1969 and released on the short-lived Happy Tiger
label.

 It was Basie's second album of Beatles' compositions following 1966's "Basie's Beatle Bag" and featured liner notes by Ringo Starr.
</item 10>
=== END ITEM id=a989891f9f13748e ===

=== ITEM id=57f9887f29ce7211 ===
Gold sentences and facts:
Sentence 0:  Game included wild boar, and red, fallow and roe deer.
  Fact 0.0: Game included wild boar.
  Fact 0.1: Game included red deer.
  Fact 0.2: Game included fallow deer.
  Fact 0.3: Game included roe deer.
Sentence 1:  This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.
  Fact 1.0: The common species is native to western Eurasia.
  Fact 1.1: The common species has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.

Retrieved items:
<item 1>
Visayan spotted deer
The Visayan spotted deer ("Rusa alfredi"), also known as the Philippine spotted deer, is a nocturnal and endangered species of deer located primarily in the
rainforests of the Visayan islands of Panay and Negros though it once roamed other islands such as Cebu, Guimaras, Leyte, Masbate, and Samar.

 It is one of three endemic deer species in the Philippines, although it was not recognized as a separate species until 1983.

 An estimated 2,500 mature individuals survived worldwide as of 1996, according to the IUCN, although it is uncertain of how many of them still survive in the
wild.

 The diet of the deer, which consists of a variety of different types of grasses, leaves, and buds within the forest, is the primary indicator of its habitat.

 Since 1991 the range of the species has severely decreased and is now almost co-extensive with that of the Visayan warty pig.
</item 1>

<item 2>
Red deer
The red deer ("Cervus elaphus") is one of the largest deer species.

 The red deer inhabits most of Europe, the Caucasus Mountains region, Asia Minor, Iran, parts of western Asia, and central Asia.

 It also inhabits the Atlas Mountains region between Morocco and Tunisia in northwestern Africa, being the only species of deer to inhabit Africa.

 Red deer have been introduced to other areas, including Australia, New Zealand, United States, Canada, Peru, Uruguay, Chile and Argentina.

 In many parts of the world, the meat (venison) from red deer is used as a food source.
</item 2>

<item 3>
Forests of Mara and Mondrem
The Forests of Mara and Mondrem were adjacent medieval forests in Cheshire, England, which in the 11th century extended to over 60 sqmi , stretching from the
Mersey in the north almost to Nantwich in the south, and from the Gowy in the west to the Weaver in the east.

 Mara and Mondrem were a hunting forest of the Norman Earls of Chester, established soon after 1071 by the first earl, Hugh d'Avranches.

 They might earlier have been an Anglo-Saxon hunting forest.

 Game included wild boar, and red, fallow and roe deer.
</item 3>

<item 4>
Indian muntjac
The Indian muntjac ("Muntiacus muntjak"), also called red muntjac and barking deer, is a common muntjac deer species in South and Southeast Asia.

 It is listed as Least Concern on the IUCN Red List.
</item 4>

<item 5>
Moose
The moose (North America) or elk (Eurasia), "Alces alces", is the largest extant species in the deer family.

 Moose are distinguished by the broad, flat (or "palmate") antlers of the males; other members of the family have antlers with a dendritic ("twig-like")
configuration.

 Moose typically inhabit boreal forests and temperate broadleaf and mixed forests of the Northern Hemisphere in temperate to subarctic climates. Hunting and
other human activities have caused a reduction in the size of the moose's range over time.

 Moose have been reintroduced to some of their former habitats.

 Currently, most moose are found in Canada, Alaska, New England, Fennoscandia, Baltic States, and Russia.

 Their diet consists of both terrestrial and aquatic vegetation.

 The most common moose predators are the gray wolf along with bears and humans. Unlike most other deer species, moose are solitary animals and do not form
herds.

 Although generally slow-moving and sedentary, moose can become aggressive and move quickly if angered or startled.

 Their mating season in the autumn features energetic fights between males competing for a female.
</item 5>

<item 6>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 6>

<item 7>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 7>

<item 8>
Fallow deer
The fallow deer ("Dama dama") is a ruminant mammal belonging to the family Cervidae.

 This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar,
Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the
United States, the Falkland Islands, and Peru.

 Some taxonomers include the rarer Persian fallow deer as a subspecies ("D. d. mesopotamica"), while others treat it as an entirely different species ("D.
mesopotamica").
</item 8>

<item 9>
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 9>

<item 10>
Sri Lankan sambar deer
The Sri Lankan sambar deer ("Rusa unicolor unicolor") is a subspecies of sambar deer that lives in Sri Lanka.

 This subspecies is one of the largest sambar deer species with the largest antlers both in size and in body proportions.

 Large males weight up to 270–280 kg.

 Sri Lankan sambar live in lowland dry forests and mountain forests.

 Large herds of sambar deer roam the Horton Plains National Park, where it is the most common large mammal.
</item 10>
=== END ITEM id=57f9887f29ce7211 ===

=== ITEM id=2bd4111665798694 ===
Gold sentences and facts:
Sentence 0: The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between 1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
  Fact 0.0: The New York – New Jersey Line War is also known as the N.J. Line War.
  Fact 0.1: The New York – New Jersey Line War refers to a series of skirmishes and raids.
  Fact 0.2: The skirmishes and raids took place for over half a century.
  Fact 0.3: The skirmishes and raids took place between 1701 and 1765.
  Fact 0.4: The skirmishes and raids took place at the disputed border between two American colonies.
  Fact 0.5: The two American colonies were the Province of New York and the Province of New Jersey.
Sentence 1: The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.
  Fact 1.0: The Province of New York existed from 1664 to 1776.
  Fact 1.1: The Province of New York was a British proprietary colony.
  Fact 1.2: The Province of New York later became a royal colony.
  Fact 1.3: The Province of New York was on the northeast coast of North America.

Retrieved items:
<item 1>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 1>

<item 2>
Province of Pennsylvania
The Province of Pennsylvania, also known as the Pennsylvania Colony, was founded in English North America by William Penn on March 4, 1681 as dictated in a
royal charter granted by King Charles II.

 The name Pennsylvania, which translates roughly as "Penn's Woods", was created by combining the Penn surname (in honor of William's father, Admiral Sir William
Penn) with the Latin word "sylvania", meaning "forest land."

 The Province of Pennsylvania was one of the two major Restoration colonies, the other being the Province of Carolina.

 The proprietary colony's charter remained in the hands of the Penn family until the American Revolution, when the Commonwealth of Pennsylvania was created and
became one of the original thirteen states.
</item 2>

<item 3>
Stamp Act Congress
The Stamp Act Congress or First Congress of the American Colonies was a meeting held between October 7 and 25, 1765 in New York City, consisting of
representatives from some of the British colonies in North America; it was the first gathering of elected representatives from several of the American colonies
to devise a unified protest against new British taxation.

 Parliament had passed the Stamp Act, which required the use of specially stamped paper for legal documents, playing cards, calendars, newspapers and dice for
virtually all business in the colonies, and was going into effect on November 1.
</item 3>

<item 4>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 4>

<item 5>
Charter colony
Charter colony is one of three classes of colonial government established in the 17th century English colonies in North America, the other classes being
proprietary colony and royal colony.

 The colonies of Rhode Island, Connecticut, and Massachusetts Bay were charter colonies.

 In a charter colony, Britain granted a charter to the colonial government establishing the rules under which the colony was to be governed.

 The charters of Rhode Island and Connecticut granted the colonists significantly more political liberty than other colonies.

 Rhode Island and Connecticut continued to use their colonial charters as their State constitutions after the American Revolution.
</item 5>

<item 6>
History of the New Jersey State Constitution
Originally, the state of New Jersey was a single British colony, the Province of New Jersey.

 After the English Civil War, Charles II assigned New Jersey as a proprietary colony to be held jointly by Sir George Carteret and John Berkeley, 1st Baron
Berkeley of Stratton.

 Eventually, the collection of land fees, or quit-rents, from colonists proved inadequate for colonial profitability.

 Sir George Carteret sold his share of the colony to the Quakers in 1673.

 Following the sale, the land was divided into East and West Jersey.

 In 1681, West Jersey adopted a constitution.

 In 1683, East Jersey adopted one as well.

 In 1702, the colonies were united again under Anne, Queen of Great Britain, and adopted a constitution in 1776.
</item 6>

<item 7>
Province of New York
The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.

 As one of the Thirteen Colonies, New York achieved independence and worked with the others to found the United States.
</item 7>

<item 8>
Proprietary colony
A proprietary colony was a type of British colony mostly in North America and the Caribbean in the 17th century.

 In the British Empire, all land belonged to the ruler, and it was his prerogative to divide.

 Therefore, all colonial properties were partitioned by royal charter into one of four types: proprietary, royal, joint stock, or covenant.

 King Charles II used the proprietary solution to reward allies and focus his own attention on Britain itself.

 He offered his friends colonial charters which facilitated private investment and colonial self-government.

 The charters made the proprietor the effective ruler, albeit one ultimately responsible to English law and the king.

 Charles II gave New Netherland to his younger brother The Duke of York, who named it New York.

 He gave an area to William Penn who named it Pennsylvania.
</item 8>

<item 9>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=2bd4111665798694 ===

=== ITEM id=161d3764c2dff301 ===
Gold sentences and facts:
Sentence 0: Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.
  Fact 0.0: Bicycling is a cycling brand.
  Fact 0.1: Bicycling is published by Rodale, Inc. in Emmaus, Pennsylvania.
Sentence 1: National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural affairs.
  Fact 1.0: National Review (NR) is an American magazine.
  Fact 1.1: National Review is semi-monthly.
  Fact 1.2: National Review is conservative.
  Fact 1.3: National Review is an editorial magazine.
  Fact 1.4: National Review focuses on news and commentary pieces on political affairs.
  Fact 1.5: National Review focuses on news and commentary pieces on social affairs.
  Fact 1.6: National Review focuses on news and commentary pieces on cultural affairs.

Retrieved items:
<item 1>
The Human Life Review
The Human Life Review is a quarterly journal published by the Human Life Foundation since 1975.

 It is devoted to explorations of life issues, primarily abortion, as well as neonaticide, medical genetics, prenatal testing, human cloning, fetal tissue
experimentation, euthanasia and assisted suicide, and also publishes articles dealing with more general questions of family and society.

 It was founded by James Patrick McFadden, formerly associate publisher of "National Review", who had also founded the Human Life Foundation, and is now edited
by his daughter, Maria McFadden.

 It was launched from the offices of "National Review", with the support of William F. Buckley.
</item 1>

<item 2>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 2>

<item 3>
Kevin D. Williamson
Kevin Daniel Williamson (born September 18, 1972) is the roving correspondent for "National Review."

 He is also the theater critic for "The New Criterion".

 He was previously deputy managing editor at "National Review".
</item 3>

<item 4>
National Review Board
The National Review Board (full name: National Review Board for the Protection of Children and Young People) is a committee created in 2002 by the United States
Conference of Catholic Bishops in order to monitor the implementation of the Charter for the Protection of Children and Young People in the wake of the clerical
abuse scandal in the United States.

 The Board was also charged with investigating the scandal, which it did in part by commissioning the John Jay College to conduct a survey of Church records in
order to define the nature and scope of the abuse committed by priests over the period between 1950 and 2002.

 The results of that survey were released in 2004 in what has come to be known as the "John Jay Report" or "Nature and Scope report."

 In parallel with the John Jay College survey the Board conducted interviews with a variety of people, both inside and outside the Church, who were well placed
to comment on the scandal, and on the basis of these interviews prepared a more broad-ranging report of its own.
</item 4>

<item 5>
Review
A review is an evaluation of a publication, service, or company such as a movie (a movie review), video game (video game review), musical composition (music
review of a composition or recording), book (book review); a piece of hardware like a car, home appliance, or computer; or an event or performance, such as a
live music concert, play, musical theater show, dance show, or art exhibition.

 In addition to a critical evaluation, the review's author may assign the work a rating to indicate its relative merit.

 More loosely, an author may review current events, trends, or items in the news.

 A compilation of reviews may itself be called a review. "

The New York Review of Books", for instance, is a collection of essays on literature, culture, and current affairs. "

National Review", founded by William F. Buckley, Jr., is an influential conservative magazine, and "Monthly Review" is a long-running socialist periodical.
</item 5>

<item 6>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 6>

<item 7>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 7>

<item 8>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
</item 8>

<item 9>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 9>

<item 10>
National Review
National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural
affairs.

 The magazine was founded by the author William F. Buckley Jr. in 1955.

 It is currently edited by Rich Lowry.
</item 10>
=== END ITEM id=161d3764c2dff301 ===

=== ITEM id=f53cfbbc583b8db7 ===
Gold sentences and facts:
Sentence 0:  His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".
  Fact 0.0: The unnamed subject's American superior was General Dwight D. Eisenhower.
  Fact 0.1: Dwight D. Eisenhower wrote that the unnamed subject was 'blunt, at times to the point of rudeness'.
Sentence 1: Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United States from 1953 until 1961.
  Fact 1.0: Dwight David 'Ike' Eisenhower was born on October 14, 1890.
  Fact 1.1: Dwight David 'Ike' Eisenhower died on March 28, 1969.
  Fact 1.2: Dwight David 'Ike' Eisenhower was an American politician.
  Fact 1.3: Dwight David 'Ike' Eisenhower was an Army general.
  Fact 1.4: Dwight David 'Ike' Eisenhower served as the 34th President of the United States.
  Fact 1.5: Dwight David 'Ike' Eisenhower served as President from 1953 until 1961.

Retrieved items:
<item 1>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 1>

<item 2>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 2>

<item 3>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 3>

<item 4>
Wendell Anderson
Wendell Richard "Wendy" Anderson (February 1, 1933 – July 17, 2016) was an American politician and the 33rd governor of Minnesota, serving from January 4, 1971,
to December 29, 1976.

 In late 1976, he resigned as governor in order to be appointed to the U.S. Senate after Senator Walter Mondale was elected Vice President of the United States.

 Anderson served in the Senate from December 30, 1976, to December 29, 1978 (after losing the 1978 Senate election to Rudy Boschwitz, he resigned a few days
before the end of his term to give Boschwitz seniority).
</item 4>

<item 5>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 5>

<item 6>
Joseph R. Anderson
Joseph Reid Anderson (February 16, 1813 – September 7, 1892) was an American civil engineer, industrialist, and soldier.

 During the American Civil War he served as a Confederate general, and his Tredegar Iron Company was a major source of munitions and ordnance for the
Confederate States Army.

 Starting with a small forge and rolling mill in the mid-1830s, It was a flourishing operation by 1843 when he leased it.

 He eventually bought the company outright in 1848 and forcefully and aggressively built Tredegar Iron Works into the South's largest and most significant iron
works.

 When the Civil War broke out he entered the Army as a Brigadier General in 1861.

 Shortly after he was wounded and then resigned from the Army returning to the iron works.

 It was the Confederacy's major (and for much of the war only) source of cannons and munitions, employing some 900 workers, most of whom slaves.

 His plant was confiscated by the United States In 1865, but returned to him in 1867 and he remained president until his death.

 Anderson was very active in local civic and political affairs.
</item 6>

<item 7>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 7>

<item 8>
Dwight D. Eisenhower
Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United
States from 1953 until 1961.

 During World War II, he was a five-star general in the United States Army and served as Supreme Commander of the Allied Expeditionary Forces in Europe.

 He was responsible for planning and supervising the invasion of North Africa in Operation Torch in 1942–43 and the successful invasion of France and Germany in
1944–45 from the Western Front.
</item 8>

<item 9>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 9>

<item 10>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 10>
=== END ITEM id=f53cfbbc583b8db7 ===

=== ITEM id=a946f533095b234a ===
Gold sentences and facts:
Sentence 0: Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.
  Fact 0.0: Victor Ronald Salva was born March 29, 1958.
  Fact 0.1: Victor Ronald Salva is an American film director.
  Fact 0.2: Victor Ronald Salva is an American screenwriter.
Sentence 1: Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.
  Fact 1.0: Emilio "El Indio" Fernández was born Emilio Fernández Romo.
  Fact 1.1: Emilio "El Indio" Fernández was born March 26, 1904.
  Fact 1.2: Emilio "El Indio" Fernández died August 6, 1986.
  Fact 1.3: Emilio "El Indio" Fernández was a Mexican film director.
  Fact 1.4: Emilio "El Indio" Fernández was a Mexican actor.
  Fact 1.5: Emilio "El Indio" Fernández was a Mexican screenwriter.

Retrieved items:
<item 1>
Columba Domínguez
Columba Domínguez Adalid (March 4, 1929 – August 13, 2014) was a Mexican actress, singer, and painter.

 Considered a crucial figure in the Golden Age of Mexican cinema, she was one of the muses of the film director Emilio Fernández, with whom she was romantically
linked for several years.

 She is remembered particularly for her performance in the film "Pueblerina" (1949), which is considered one of the jewels of the Mexican Cinema.
</item 1>

<item 2>
Jaime Fernández (actor)
Jaime Fernández Reyes (December 6, 1937 – April 15, 2005) was a Mexican actor.

 Over his career, he won 3 Silver Ariel awards - the Mexican equivalent of the Oscar - including one for what is arguably his best-known role, playing Friday in
Luis Buñuel's "Robinson Crusoe".

 He appeared in over 200 films and served as the General Secretary of the Mexican actors' union for 11 years.

 His older brother was actor/director Emilio Fernández.
</item 2>

<item 3>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 3>

<item 4>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 4>

<item 5>
Comedy in the Golden Age of Mexican cinema
Sadly, it was not as successful as many directors and film maker though it would be.

 Many failed to correctly synchronize the audio with the picture.

 Although to some extent it may have been unsuccessful, it is agreeable that it helped revolutionize film in Latin America.

 Mexico’s film industry takes its height post “El Cine Sonoro” to what is known as “The Golden Age”.

 Many film members consider this one of the most successful eras in the history of Mexican cinema.

 Many recognize the films of this era when they see black and white.

 It is also important to note that the films also portrayed social issues.

 Some of the most influential directors and actors are Emilio Fernández, Raul De Anda, Roberto B. Cervenna, Cantinflas, Vitola, Tin Tan, etc.

 The focus of this article will be on some of top grossing comedian actors of the time Cantinflas, Tin Tan, Clavillazo and others.
</item 5>

<item 6>
Valerio Fioravanti
Giuseppe Valerio "Giusva" Fioravanti (born 28 March 1958) is an Italian who, with Francesca Mambro, was a leading figure in a far right terrorist group "Nuclei
Armati Rivoluzionari" ("Armed Revolutionary Nuclei" or NAR).

 Fioravanti appeared in films and television from a young age, and in his early teens was the most famous child in Italy.

 He and Mambro were fugitives wanted for terrorist offences by their early twenties, they spent a further period on the run while wanted for the Bologna
bombing.

 Both were captured after gunfights with police.

 Although Fioravanti, Mambro and a third NAR mamber were convicted of the bombing, there were those who thought that a higher level of the conspiracy was never
uncovered.

 Fioravanti and Mambro admit carrying out terrorist murders, but have always denied any involvement in the Bologna station bomb.
</item 6>

<item 7>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 7>

<item 8>
Fernando Fernández (actor)
Fernando Fernández "El Crooner de México" (1916–1999) was a Mexican actor and singer.

 He was born on November 9, 1916, in Monterrey, Mexico, and died in Mexico City, Mexico, on November 24, 1999.

 Fernando was the son of Eloisa Reyes.

 He was brother of the famous film director Emilio Fernández and actor Jaime Fernández.

 He was married to singer Lupita Palomera, who died in 2008.

 Fernando Fernández was known as "the Crooner of Mexico."
</item 8>

<item 9>
Emilio Fernández
Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.

 He was one of the most prolific film directors of the Golden Age of Mexican cinema in the 1940s and 1950s.

 He is best known for his work as director of the film "Maria Candelaria" (1944), which won the Palme d'Or award at the 1946 Cannes Film Festival.

 As an actor, he worked in numerous film productions in Mexico and in Hollywood.
</item 9>

<item 10>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 10>
=== END ITEM id=a946f533095b234a ===

=== ITEM id=86b24355cfd3e34e ===
Gold sentences and facts:
Sentence 0:  He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.
  Fact 0.0: He is mainly known for his spells at Real Sociedad.
  Fact 0.1: His offensive partnership with Nihat Kahveci at Real Sociedad was one of the best in Spain.
Sentence 1: Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
  Fact 1.0: Nihat Kahveci was born on 23 November 1979.
  Fact 1.1: Nihat Kahveci is a Turkish former footballer.
  Fact 1.2: Nihat Kahveci played as a forward.

Retrieved items:
<item 1>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 1>

<item 2>
Lippo Hertzka
Lippo Hertzka (19 November 1904 – 14 March 1951) was a Hungarian football player and manager.

 He played for Essener Turnerbund, MTK Budapest and Real Sociedad.

 After retiring, he coached seven teams, including Real Sociedad and Real Madrid, a team which he coached for 2 years (1930–1932) and led to an undefeated La
Liga championship during the 1931-32 season, which meant the first La Liga title for the "white" squad.

 He also won two league titles in Portugal for Benfica.
</item 2>

<item 3>
C.D. Real Sociedad
Club Deportivo Real Sociedad, commonly known as Real Sociedad (] ), is a Honduran football club based on Tocoa, Colón, Honduras.
</item 3>

<item 4>
Real Sociedad
Real Sociedad de Fútbol, S.A.D., more commonly referred to as Real Sociedad (] ; "Royal Society") or La Real, is a Spanish football club based in the city of
San Sebastián, Basque Country, founded on 7 September 1909.

 It plays its home matches at the 32,000-capacity Anoeta Stadium.

 Real Sociedad won the Liga title in 1980–81 and 1981–82, and last finished runners-up in 2002–03.

 The club has also won the Copa del Rey twice, in 1909 and 1987.

 It contests the Basque derby against rivals Athletic Bilbao.

 Real Sociedad were founder members of La Liga in 1928, and its longest spell in the top flight was for 40 seasons, from 1967 to 2007.
</item 4>

<item 5>
Nihat Kahveci
Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
</item 5>

<item 6>
Darko Kovačević
Darko Kovačević (] ; Serbian Cyrillic: ; born 18 November 1973) is a Serbian former footballer who played as a forward.

 He began his career in Serbia with Proleter Zrenjanin and subsequently played for Red Star Belgrade, with whom he won a Yugoslav League title and two Yugoslav
Cups.

 His prolific performances earned him a move to Premier League side Sheffield Wednesday, although his time in England was less successful.

 He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.

 Kovačević also had positive spells with Italian club Juventus and Greek side Olympiacos.

 At international level, he represented Yugoslavia at the 1998 FIFA World Cup and at UEFA Euro 2000.
</item 6>

<item 7>
Adnan Januzaj
Adnan Januzaj (] ; born 5 February 1995) is a Belgian professional footballer who plays as a winger for Spanish club Real Sociedad.

 Born and raised in Brussels, he began his career with Anderlecht before joining Manchester United in 2011 at the age of 16.

 Januzaj broke into the Manchester United first-team under manager David Moyes during the 2013–14 season, but struggled for opportunities under Moyes'
successors Louis van Gaal and José Mourinho, and had loan spells at Borussia Dortmund and Sunderland before joining Real Sociedad in July 2017.

 Januzaj made his full international debut in 2014 and later that year played for Belgium at the World Cup.
</item 7>

<item 8>
2013–14 Real Sociedad season
The 2013–14 season was Real Sociedad's 67th season in La Liga.

 Real Sociedad finished 7th in the league and reached the semifinals of the Copa del Rey.

 The Basque failed to make it out of the group stages of the UEFA Champions League.
</item 8>

<item 9>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 9>

<item 10>
Ivan Toplak
Ivan Toplak (Serbian Cyrillic: Ивaн Toплaк; born September 21, 1931 in Belgrade) is a former Yugoslavian football player and manager.

 He earned 1 cap for Yugoslavia.

 He started his career at Branik in Maribor, where his family moved in 1943 from Belgrade.
</item 10>
=== END ITEM id=86b24355cfd3e34e ===

=== ITEM id=4fe3591722b68689 ===
Gold sentences and facts:
Sentence 0:  He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".
  Fact 0.0: This person is most well known for providing the singing voice of Tony in the 1961 film musical "West Side Story".
  Fact 0.1: Richard Beymer played Tony onscreen in the 1961 film musical "West Side Story".
Sentence 1:  The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the most wins for a movie musical.
  Fact 1.0: West Side Story was nominated for 11 Academy Awards.
  Fact 1.1: West Side Story won 10 Academy Awards.
  Fact 1.2: West Side Story won Best Picture.
  Fact 1.3: West Side Story won a special award for Robbins.
  Fact 1.4: West Side Story became the record holder for the most wins for a movie musical.

Retrieved items:
<item 1>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 1>

<item 2>
Graph query row:
work: West Side Story
award: Academy Awards
awards_won: 10
</item 2>

<item 3>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 3>

<item 4>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 4>

<item 5>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 5>

<item 6>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 6>

<item 7>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 7>

<item 8>
Jimmy Bryant (singer)
James Howard Bryant (born June 2, 1929) is a singer, arranger and composer.

 He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".

 While he received no screen credit, he states that Beymer was "a nice guy, and every time he did an interview he would mention my name."

 He also sang for James Fox in the 1967 film musical "Thoroughly Modern Millie", and sang in "The Telephone Hour" number in "Bye Bye Birdie".

 He also sang in the group that performed the theme song of the TV series "Batman".
</item 8>

<item 9>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 9>

<item 10>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 10>
=== END ITEM id=4fe3591722b68689 ===

=== ITEM id=46a387aea0bd925d ===
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
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 1>

<item 2>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 2>

<item 3>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 3>

<item 4>
Eenasul Fateh (Aladin, b. 1959) is a Bangladeshi-British magician and cultural practitioner. “Abdāl” denotes saintly figures in Islamic mysticism. Dawud Tai (d.
777–782) was a Sufi scholar; Muhammad ibn Thaleb a 15th-century Persian physician. Ibn al-Khatib (1313–1374) was a Granada-born Moroccan polymath. Da’ud Abu
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
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 6>

<item 7>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 7>

<item 8>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 8>

<item 9>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 9>

<item 10>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 10>
=== END ITEM id=46a387aea0bd925d ===

=== ITEM id=aed60dd7c35334ed ===
Gold sentences and facts:
Sentence 0: The City Heiress is a play by Aphra Behn produced in 1682.
  Fact 0.0: The City Heiress is a play by Aphra Behn.
  Fact 0.1: The City Heiress was produced in 1682.
Sentence 1: Aphra Behn ( ; 14 December 1640?
  Fact 1.0: Aphra Behn's birth date is listed as 14 December 1640, marked with a question mark as uncertain.
Sentence 2:  (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.
  Fact 2.0: The 14 December 1640 date for Aphra Behn is a baptismal date.
  Fact 2.1: Aphra Behn died on 16 April 1689.
  Fact 2.2: Aphra Behn was a British playwright.
  Fact 2.3: Aphra Behn was a poet.
  Fact 2.4: Aphra Behn was a translator.
  Fact 2.5: Aphra Behn was a fiction writer.
  Fact 2.6: Aphra Behn was from the Restoration era.

Retrieved items:
<item 1>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 1>

<item 2>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 2>

<item 3>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 3>

<item 4>
What the Butler Saw (play)
What the Butler Saw is a farce written by the English playwright Joe Orton.

 It was premièred at the Queen's Theatre in London on 5 March 1969.

 It was Orton's final play and the second to be performed after his death, following "Funeral Games" in 1968.
</item 4>

<item 5>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 5>

<item 6>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 6>

<item 7>
The City Heiress
The City Heiress is a play by Aphra Behn produced in 1682.

 The play conforms to the general rules of Restoration comedy, but it also keeps Behn's own highly Royalist political point of view.
</item 7>

<item 8>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 8>

<item 9>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 9>

<item 10>
Graph query row:
person: Aphra Behn
death_date: 16 April 1689
relationship: CREATED_BY
creative_work: The City Heiress
</item 10>
=== END ITEM id=aed60dd7c35334ed ===

=== ITEM id=b5b36ad5e1edd579 ===
Gold sentences and facts:
Sentence 0: Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.
  Fact 0.0: Ferdi Taygan was born on December 5, 1956
  Fact 0.1: Ferdi Taygan is a former professional tennis player
  Fact 0.2: Ferdi Taygan is from the United States
Sentence 1: Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.
  Fact 1.0: Mahesh Shrinivas Bhupathi was born on 7 June 1974
  Fact 1.1: Mahesh Shrinivas Bhupathi is a retired Indian professional tennis player

Retrieved items:
<item 1>
The material compiles tennis results and player biographies. Sherwood Stewart and Ferdi Taygan won the 1982 French Open doubles and finished runner-up at the
Volvo Masters and U.S. Pro Indoor; Stewart later partnered others. Jorge Lozano, a Mexican doubles No. 4, won nine doubles titles, including two French Open
mixed doubles crowns, and captained Mexico’s Davis Cup team. Brazilian Carlos Fernandes reached the 1961 French Open quarterfinals. Argentine Javier Fr
</item 1>

<item 2>
Mahesh Bhupathi Tennis Academy
Mahesh Bhupathi Tennis Academy is a tennis training facility started by Indian tennis player Mahesh Bhupathi.

 The Academy has facilities across 9 states in India as well as in the UAE.
</item 2>

<item 3>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 3>

<item 4>
Ferdi Taygan was a Turkish-American former pro. Indian star Mahesh Bhupathi won major mixed-doubles milestones, founded the IPTL and a tennis academy in India
and UAE, and captained India’s Davis Cup team; he and Leander Paes won the 2004 Canada Masters doubles. Other profiles include Latvian Ernests Gulbis (six ATP
titles, French Open semifinalist, former No. 10), Austrian wheelchair champion Martin Legner, former world
</item 4>

<item 5>
1982 Volvo Masters – Doubles
Peter Fleming and John McEnroe won in the final 6–4, 6–3 against Sherwood Stewart and Ferdi Taygan.
</item 5>

<item 6>
Mahesh Bhupathi
Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.

 In 1997, he became the first Indian to win a Grand Slam tournament (with Rika Hiraki).

 With his win at the Australian Open mixed doubles in 2006, he joined the elite group of eight tennis players who have achieved a career Grand Slam in mixed
doubles.

 He is also the founder of International Premier Tennis League.

 In December 2016, Bhupathi was appointed as India's next non-playing Davis Cup captain and took over the reins from Anand Amritraj in February 2017.
</item 6>

<item 7>
Ferdi Taygan
Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.

 He is of Turkish descent.
</item 7>

<item 8>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 8>

<item 9>
1983 Donnay Indoor Championships – Doubles
Pavel Složil and Sherwood Stewart were the defending champions, but Složil did not participate this year.

 Stewart partnered Ferdi Taygan, losing in the first round.
</item 9>

<item 10>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 10>
=== END ITEM id=b5b36ad5e1edd579 ===
