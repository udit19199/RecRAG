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

=== ITEM id=39b0a98655fd3dec ===
Gold sentences and facts:
Sentence 0: Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years elapsed 304 123 47 50
  Fact 0.0: The starting value is 500 million.
  Fact 0.1: The population values, in billions, are 0.5 and 8.
  Fact 0.2: The year values are 1500, 1804, 1927 and 2024.
  Fact 0.3: The years elapsed values are 304, 123, 47 and 50.

Retrieved items:
<item 1>
World population
The vertical axis is logarithmic and is in millions of people . Population size fluctuates at differing rates in differing regions . Nonetheless , population
growth is the long - standing trend on all inhabited continents , as well as in most individual states . During the 20th century , the global population saw its
greatest increase in known history , rising from about 1.6 billion in 1900 to over 6 billion in 2000 . A number of factors contributed to this increase ,
including the lessening of the mortality rate in many countries by improved sanitation and medical advances , and a massive increase in agricultural
productivity attributed to the Green Revolution . In 2000 , the United Nations estimated that the world 's population was growing at an annual rate of 1.14 % (
equivalent to around 75 million people ) , down from a peak of 88 million per year in 1989 . By 2000 , there were approximately ten times as many people on
Earth as there had been in 1700 .
</item 1>

<item 2>
World population
Milestones by the billions ( edit ) Main article : World population milestones World population milestones in billions Population 5 6 7 8 9 Year 1804 1927 1960
1999 2011 2024 2042 Years elapsed -- -- 123 33 14 13 12 12 13 18 It is estimated that the world population reached one billion for the first time in 1804 . It
was another 123 years before it reached two billion in 1927 , but it took only 33 years to reach three billion in 1960 . Thereafter , the global population
reached four billion in 1974 , five billion in 1987 , six billion in 1999 and , according to the United States Census Bureau , seven billion in March 2012 . The
United Nations , however , estimated that the world population reached seven billion in October 2011 . According to current projections , the global population
will reach eight billion by 2024 , and will likely reach around nine billion by 2042 . Alternative scenarios for 2050 range from a low of 7.4 billion to a high
of more than 10.6 billion .
</item 2>

<item 3>
World population
The transition from hyperbolic growth to slower rates of growth is related to the demographic transition . According to the Russian demographer Sergey Kapitsa ,
the world population grew between 67,000 BC and 1965 according to the following formula : N = C τ arccot ⁡ T 0 − T τ ( \ displaystyle N = ( \ frac ( C ) ( \ tau
) ) \ operatorname ( arccot ) ( \ frac ( T_ ( 0 ) - T ) ( \ tau ) ) ) where N is current population T is the current year C = ( 1.86 ± 0.01 ) 10 T = 2007 ± 1 τ
( \ displaystyle \ tau ) = 42 ± 1 Years for world population to double ( edit ) According to linear interpolation and extrapolation of UNDESA population
estimates , the world population has doubled , or will double , in the years listed in the tables below ( with two different starting points ) . During the 2nd
millennium , each doubling took roughly half as long as the previous doubling , fitting the hyperbolic growth model mentioned above .
</item 3>

<item 4>
World population
The world population has grown by over four billion since the beginning of the Green Revolution , but food production has so far kept pace with population
growth . Most scholars believe that , without the Revolution , there would be greater levels of famine and malnutrition than the UN presently documents .
However , neo-Malthusians point out that fossil fuels provided the energy for the Green Revolution , in the form of natural gas - derived fertilizers , oil -
derived pesticides , and hydrocarbon - fueled irrigation , and that many crops have become so genetically uniform that a crop failure in any one country could
potentially have global repercussions . In 2004 , a meta - analysis of 70 quantitative studies estimating a sustainable limit to the world population generated
a meta - estimate of 7.7 billion people . Graph of the global human population from 10,000 BC to AD 2000 , from the US Census Bureau . The graph shows the
extremely rapid growth in the world population that has taken place since the 18th century . The human population has been growing rapidly since the mid-1800s .
</item 4>

<item 5>
World population
2 % ) 4,051 ( 37.6 % ) 644 ( 6.0 % ) 747 ( 6.9 % ) 510 ( 4.7 % ) 69 ( 0.6 % ) 2100 10,854 4,712 ( 43.4 % ) 4,185 ( 38.6 % ) 639 ( 5.9 % ) 736 ( 6.8 % ) 513 (
4.7 % ) 70 ( 0.6 % ) Mathematical approximations ( edit ) In 1975 , Sebastian von Hoerner proposed a formula for population growth which represented hyperbolic
growth with an infinite population in 2025 . The hyperbolic growth of the world population observed until the 1970s was later correlated to a non-linear second
order positive feedback between demographic growth and technological development .
</item 5>

<item 6>
World population
000 BC < 0.015 0 0 10,000 BC 8000 BC 5 6500 BC 5 5000 BC 5 4000 BC 7 3000 BC 14 2000 BC 27 1000 BC 50 7 33 9 500 BC 100 14 66 16 AD 1 200 23 141 28 1000 400 70
269 50 8 1500 458 86 243 84 39 1600 580 114 339 111 10 1700 682 106 436 125 10 1750 791 106 502 163 16 1800 1,000 107 656 203 24 7 1850 1,262 111 809 276 38 26
1900 1,650 133 947 408 74 82 6 1950 2,525 229 1,394 549 169 172 12.7 1955 2,758 254 1,534 577 193 187 14.
</item 6>

<item 7>
World population
Early estimates for the population of the world date to the 17th century : William Petty in 1682 estimated world population at 320 million ( modern estimates
ranging close to twice this number ) ; by the late 18th century , estimates ranged close to one billion ( consistent with modern estimates ) . More refined
estimates , broken down by continents , were published in the first half of the 19th century , at 600 to 1000 million in the early 1800s and at 800 to 1000
million in the 1840s . Antiquity and Middle ages ( edit ) Main articles : Classical demography and Medieval demography Estimates of the population of the world
at the time agriculture emerged in around 10,000 BC have ranged between 1 million and 15 million . Even earlier , genetic evidence suggests humans may have gone
through a population bottleneck of between 1,000 and 10,000 people about 70,000 BC , according to the Toba catastrophe theory . By contrast , it is estimated
that around 50 -- 60 million people lived in the combined eastern and western Roman Empire in the 4th century AD .
</item 7>

<item 8>
World population
Projected figures vary depending on underlying statistical assumptions and the variables used in projection calculations , especially the fertility variable .
Long - range predictions to 2150 range from a population decline to 3.2 billion in the `` low scenario '' , to `` high scenarios '' of 24.8 billion . One
extreme scenario predicted a massive increase to 256 billion by 2150 , assuming the global fertility rate remained at its 1995 level of 3.04 children per woman
; however , by 2010 the global fertility rate had declined to 2.52 . There is no estimation for the exact day or month the world 's population surpassed one or
two billion . The points at which it reached three and four billion were not officially noted , but the International Database of the United States Census
Bureau placed them in July 1959 and April 1974 respectively . The United Nations did determine , and commemorate , the `` Day of 5 Billion '' on July 11 , 1987
, and the `` Day of 6 Billion '' on October 12 , 1999 . The Population Division of the United Nations declared the `` Day of 7 Billion '' to be October 31 ,
2011 .
</item 8>

<item 9>
World population
Improved material conditions led to the population of Britain increasing from 10 million to 40 million in the 19th century . The population of the United
Kingdom reached 60 million in 2006 . The United States saw its population grow from around 5.3 million in 1800 to 106 million in 1920 , exceeding 307 million in
2010 . The first half of the 20th century in Imperial Russia and the Soviet Union was marked by a succession of major wars , famines and other disasters which
caused large - scale population losses ( approximately 60 million excess deaths ) . After the collapse of the Soviet Union , Russia 's population declined
significantly -- from 150 million in 1991 to 143 million in 2012 -- but by 2013 this decline appeared to have halted . Many countries in the developing world
have experienced extremely rapid population growth since the early 20th century , due to economic development and improvements in public health . China 's
population rose from approximately 430 million in 1850 to 580 million in 1953 , and now stands at over 1.3 billion .
</item 9>

<item 10>
World population
However , after 2024 , it is unlikely that there will be another doubling of the global population in the 21st century . Historic chart showing the periods of
time the world population has taken to double , from 1700 to 2000 Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years
elapsed 304 123 47 50 Starting at 375 million Population ( in billions ) 0.375 0.75 1.5 6 Year 1171 1715 1881 1960 1999 Years elapsed 544 166 79 39
Overpopulation ( edit ) Main articles : Human overpopulation and Malthusian catastrophe Predictions of scarcity ( edit ) Greater Los Angeles lies on a coastal
mediterranean savannah with a small watershed that is able to support at most one million people on its own water ; as of 2015 , the area has a population of
over 18 million . Researchers predict that similar cases of resource scarcity will grow more common as the world population increases .
</item 10>
=== END ITEM id=39b0a98655fd3dec ===

=== ITEM id=1d3211f83fd76250 ===
Gold sentences and facts:
Sentence 0: The story of Shakespeare 's Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th - century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time . In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century .
  Fact 0.0: The story of Shakespeare's Hamlet was derived from the legend of Amleth.
  Fact 0.1: The legend of Amleth was preserved by the 13th-century chronicler Saxo Grammaticus in his Gesta Danorum.
  Fact 0.2: The legend of Amleth was subsequently retold by the 16th-century scholar François de Belleforest.
  Fact 0.3: Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur-Hamlet.
  Fact 0.4: Some scholars believe Shakespeare himself wrote the Ur-Hamlet and later revised it to create the version of Hamlet we have today.
  Fact 0.5: Shakespeare almost certainly wrote his version of the title role for his fellow actor Richard Burbage.
  Fact 0.6: Richard Burbage was the leading tragedian of Shakespeare's time.
  Fact 0.7: In the 400 years since the role of Hamlet was created, it has been performed by numerous highly acclaimed actors.
  Fact 0.8: The role of Hamlet has been performed by highly acclaimed actors in each successive century for 400 years.

Retrieved items:
<item 1>
Hamlet
London , 1776 The play was revived early in the Restoration . When the existing stock of pre-civil war plays was divided between the two newly created patent
theatre companies , Hamlet was the only Shakespearean favourite that Sir William Davenant 's Duke 's Company secured . It became the first of Shakespeare 's
plays to be presented with movable flats painted with generic scenery behind the proscenium arch of Lincoln 's Inn Fields Theatre . This new stage convention
highlighted the frequency with which Shakespeare shifts dramatic location , encouraging the recurrent criticism of his failure to maintain unity of place . In
the title role , Davenant cast Thomas Betterton , who continued to play the Dane until he was 74 . David Garrick at Drury Lane produced a version that adapted
Shakespeare heavily ; he declared : `` I had sworn I would not leave the stage till I had rescued that noble play from all the rubbish of the fifth act . I have
brought it forth without the grave - digger 's trick , Osrick , & the fencing match '' . The first actor known to have played Hamlet in North America is Lewis
Hallam . Jr.
</item 1>

<item 2>
Hamlet
Hamlet - wikipedia Hamlet Jump to : navigation , search This article is about the play by William Shakespeare . For other uses , see Hamlet ( disambiguation ) .
The American actor Edwin Booth as Hamlet , ca . 1870 The Tragedy of Hamlet , Prince of Denmark , often shortened to Hamlet ( / ˈhæmlɪt / ) , is a tragedy
written by William Shakespeare at an uncertain date between 1599 and 1602 . Set in Denmark , the play dramatises the revenge Prince Hamlet is called to wreak
upon his uncle , Claudius , by the ghost of Hamlet 's father , King Hamlet . Claudius had murdered his own brother and seized the throne , also marrying his
deceased brother 's widow . Hamlet is Shakespeare 's longest play , and is considered among the most powerful and influential works of world literature , with a
story capable of `` seemingly endless retelling and adaptation by others '' .
</item 2>

<item 3>
Hamlet
He is dethroned by the spectre of an actor , and we shall never be able to keep the usurper out of our dreams . Maurice Maeterlinck in La Jeune Belgique ( 1890
) . Shakespeare 's day to the Interregnum ( edit ) Shakespeare almost certainly wrote the role of Hamlet for Richard Burbage . He was the chief tragedian of the
Lord Chamberlain 's Men , with a capacious memory for lines and a wide emotional range . Judging by the number of reprints , Hamlet appears to have been
Shakespeare 's fourth most popular play during his lifetime -- only Henry IV Part 1 , Richard III and Pericles eclipsed it . Shakespeare provides no clear
indication of when his play is set ; however , as Elizabethan actors performed at the Globe in contemporary dress on minimal sets , this would not have affected
the staging . Firm evidence for specific early performances of the play is scant .
</item 3>

<item 4>
Hamlet
It was probably one of Shakespeare 's most popular works during his lifetime , and still ranks among his most performed , topping the performance list of the
Royal Shakespeare Company and its predecessors in Stratford - upon - Avon since 1879 . It has inspired many other writers -- from Johann Wolfgang von Goethe and
Charles Dickens to James Joyce and Iris Murdoch -- and has been described as `` the world 's most filmed story after Cinderella '' . The story of Shakespeare 's
Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th
- century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars
believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title
role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time .
</item 4>

<item 5>
Meres’s omission of *Hamlet* suggests it was not yet written, though the “little eyases” allusion and War of the Theatres support a 1601 date; Katherine
Duncan-Jones favors 1600–01 but reads the allusion as playful praise of John Marston. Shakespeare likely wrote Hamlet for Richard Burbage, and the play was
highly popular. Documented performances include Sierra Leone in 1607, Germany soon after Shakespeare’s death, and royal court
</item 5>

<item 6>
Hamlet
In stark contrast to earlier opulence , William Poel 's 1881 production of the Q1 text was an early attempt at reconstructing the Elizabethan theatre 's
austerity ; his only backdrop was a set of red curtains . Sarah Bernhardt played the prince in her popular 1899 London production . In contrast to the ``
effeminate '' view of the central character that usually accompanied a female casting , she described her character as `` manly and resolute , but nonetheless
thoughtful ... ( he ) thinks before he acts , a trait indicative of great strength and great spiritual power '' . In France , Charles Kemble initiated an
enthusiasm for Shakespeare ; and leading members of the Romantic movement such as Victor Hugo and Alexandre Dumas saw his 1827 Paris performance of Hamlet ,
particularly admiring the madness of Harriet Smithson 's Ophelia . In Germany , Hamlet had become so assimilated by the mid-19th century that Ferdinand
Freiligrath declared that `` Germany is Hamlet '' . From the 1850s , the Parsi theatre tradition in India transformed Hamlet into folk performances , with
dozens of songs added .
</item 6>

<item 7>
Hamlet
Jr. , in the American Company 's production in Philadelphia in 1759 . David Garrick 's iconic hand gesture expresses Hamlet 's shock at the first sight of the
ghost ( artist : unknown ) . John Philip Kemble made his Drury Lane debut as Hamlet in 1783 . His performance was said to be 20 minutes longer than anyone else
's , and his lengthy pauses provoked the suggestion by Richard Brinsley Sheridan that `` music should be played between the words '' . Sarah Siddons was the
first actress known to play Hamlet ; many women have since played him as a breeches role , to great acclaim . In 1748 , Alexander Sumarokov wrote a Russian
adaptation that focused on Prince Hamlet as the embodiment of an opposition to Claudius 's tyranny -- a treatment that would recur in Eastern European versions
into the 20th century . In the years following America 's independence , Thomas Apthorpe Cooper , the young nation 's leading tragedian , performed Hamlet among
other plays at the Chestnut Street Theatre in Philadelphia , and at the Park Theatre in New York .
</item 7>

<item 8>
Hamlet
In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century . Three different early
versions of the play are extant : the First Quarto ( Q1 , 1603 ) ; the Second Quarto ( Q2 , 1604 ) ; and the First Folio ( F1 , 1623 ) . Each version includes
lines and entire scenes missing from the others . The play 's structure and depth of characterisation have inspired much critical scrutiny . One such example is
the centuries - old debate about Hamlet 's hesitation to kill his uncle , which some see as merely a plot device to prolong the action , but which others argue
is a dramatisation of the complex philosophical and ethical issues that surround cold - blooded murder , calculated revenge , and thwarted desire . More
recently , psychoanalytic critics have examined Hamlet 's unconscious desires , while feminist critics have re-evaluated and attempted to rehabilitate the often
maligned characters of Ophelia and Gertrude . Contents ( hide ) 1 Characters 2 Plot 2.1 Act I 2.2 Act II 2.3 Act III 2.
</item 8>

<item 9>
After the Restoration, Hamlet became the only Shakespeare play secured by Davenant’s Duke’s Company and the first staged with movable scenery, prompting
criticism of its shifting locations. Thomas Betterton played Hamlet for decades; David Garrick later cut the gravedigger, Osric, and fencing match. Lewis Hallam
Jr. was the first known North American Hamlet (1759). John Philip Kemble’s lengthy performance and Sarah Siddons’s pioneering portrayal as Hamlet were notable.
Russian adaptations
</item 9>

<item 10>
Shakespeare’s *Hamlet* was likely popular in his lifetime and remains among the most performed plays, leading the Royal Shakespeare Company’s
Stratford-upon-Avon performance list since 1879. It inspired writers including Goethe, Dickens, Joyce, and Murdoch and is among the world’s most filmed stories.
Its plot derives from Saxo Grammaticus’s Amleth legend, retold by François de Belleforest. Shakespeare may also have adapted the earlier *Ur-Hamlet* and
</item 10>
=== END ITEM id=1d3211f83fd76250 ===

=== ITEM id=d195eadda113b55c ===
Gold sentences and facts:
Sentence 0: Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and chauffeur , Mr. George Swane , 19 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 294 and cook , Miss Amelia Mary `` Mildred '' Brown 18 London , England , UK Southampton Montreal , Quebec , Canada 11 Allison , Mrs. Bessie Waldo ( née Daniels ) 25 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Sarah Daniels 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 8 Allison , Miss Helen Loraine Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Allison , Master Hudson Trevor 0.11 ! 11 mo . Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 11 and nurse , Miss Alice Catherine Cleaver 22 London , England , UK Southampton Montreal , Quebec , Canada 11 Anderson , Mr. Harry 47 New York City , New York , US Southampton New York City , New York , US Andrews , Miss Kornelia Theodosia 62 Hudson , New York , US Cherbourg Hudson , New York , US 10 Andrews , Mr. Thomas 39 Belfast , Ireland , UK Belfast New York City Appleton , Mrs. Charlotte Lane ( née Lamson ) 53 New York City Southampton New York City Artagaveytia , Mr. Ramon 71 Buenos Aires , Argentina Cherbourg New York City 22 Astor , Colonel John Jacob IV 47 New York City Cherbourg New York City 124 Astor , Mrs. Madeleine Talmage ( née Force ) 18 New York City Cherbourg New York City and maid , Miss Rosalie Bidois 46 New York City Cherbourg New York City and nurse , Miss Caroline Louise Endres 39 New York City Cherbourg New York City Aubart , Mrs. Léontine Pauline 24 Paris , France Cherbourg New York City 9 and maid , Miss Emma Sägesser 24 Paris , France Cherbourg New York City 9 Barkworth , Mr. Algernon Henry Wilson 47 Hessle , East Yorkshire , England , UK Southampton New York City Baumann , Mr. John D . 60 New York City Cherbourg New York City Baxter , Mrs. Hélène ( née de Lanaudière - Chaput ) 50 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Baxter , Mr. Quigg Edmond 24 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada Beattie , Mr. Thomson 36 Fergus , Ontario , Canada Southampton Fergus , Ontario , Canada 331 Beckwith , Mr. Richard Leonard 37 New York City Southampton New York City 5 Beckwith , Mrs. Sallie ( née Monypeny ) 46 New York City Southampton New York City 5 Behr , Mr. Karl Howell 26 New York City Cherbourg New York City 5 Birnbaum , Mr. Jakob 24 Antwerp , Belgium Cherbourg San Francisco , California , US 148 Bishop , Mr. Dickinson H. `` Dick '' 25 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Bishop , Mrs. Helen ( née Walton ) 19 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Björnström - Steffanson , Mr. Mauritz Håkan 28 Stockholm , Sweden Southampton Washington , D.C. , US Blackwell , Mr. Stephen Weart 45 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Blank , Mr. Henry 39 Glen Ridge , New Jersey , US Cherbourg Glen Ridge , New Jersey , US 7 Bonnell , Miss Elizabeth 61 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Bonnell , Miss Caroline 30 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Borebank , Mr. John James 42 London , England , UK Southampton Toronto , Ontario , Canada Bowerman , Miss Elsie Edith 22 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Brady , Mr. John Bertram 41 Pomeroy , Washington , US Southampton Pomeroy , Washington , US Brandeis , Mr. Emil 48 Omaha , Nebraska , US Cherbourg Omaha , Nebraska , US 208 Brereton , Mr. George Andrew ( alias George A. Brayton ) 37 Los Angeles , US Southampton Los Angeles , California , US 9 Brewe , Dr. Arthur Jackson 45 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US Brown , Mrs. Caroline Lane ( née Lamson ) 59 Belmont , Massachusetts , US Southampton Belmont , Massachusetts , US Brown , Mrs. Margaret ( née Tobin ) 44 Denver , Colorado , US Cherbourg Denver , Colorado , US 6 Bucknell , Mrs. Emma Eliza ( née Ward ) 59 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 and maid , Miss Albina Bazzani 36 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 Butt , Major Archibald Willingham 46 Washington , D.C. , US Southampton Washington , D.C. , US Calderhead , Mr. Edward Pennington 42 New York City Southampton New York City 5 Candee , Mrs. Helen Churchill ( née Hungerford ) 53 Washington , D.C. , US Cherbourg Washington , D.C. , US 6 Cardeza , Mrs. Charlotte Wardle ( née Drake ) 58 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and maid , Miss Annie Moore Ward 38 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Cardeza , Mr. Thomas Drake Martinez 36 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and valet , Mr. Gustave J. Lesueur 35 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Carlsson , Mr. Frans Olof 33 New York City Southampton New York City Carrau , Mr. Francisco M . 31 Montevideo , Uruguay Southampton Montevideo , Uruguay Carrau , Mr. José Pedro 17 Montevideo , Uruguay Southampton Montevideo , Uruguay Carter , Mr. William Ernest 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and valet , Mr. Alexander Cairns 28 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and chauffeur , Mr. Charles Augustus Aldworth 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Mrs. Lucile ( née Polk ) 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and maid , Miss Auguste Serreplan 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Miss Lucile Polk 13 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Master William Thornton II 11 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Case , Mr. Howard Brown 49 Ascot , Berkshire , England , UK Southampton Rochester , New York , US Cassebeer , Mrs. Eleanor Genevieve ( née Fosdick ) 36 New York City Cherbourg New York City 5 Cavendish , Mr. Tyrell William 36 London , England , UK Southampton New York City 172 Cavendish , Mrs. Julia Florence ( née Siegel ) 25 London , England , UK Southampton New York City 6 and maid , Miss Ellen `` Nellie '' Barber 26 London , England , UK Southampton New York City 6 Chaffee , Mr. Hubert Fuller 46 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chaffee , Mrs. Carrie Constance ( née Toogood ) 47 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chambers , Mr. Norman Campbell 27 New York City Southampton New York City 5 Chambers , Mrs. Bertha ( née Griggs ) 32 New York City Southampton New York City 5 Cherry , Miss Gladys 30 London , England , UK Southampton Vancouver , British Columbia , Canada 8 Chevré , Mr. Paul Romaine Marie Léonce 45 Paris , France Cherbourg Ottawa , Ontario , Canada 7 Chibnall , Mrs. Edith Martha Bowerman ( née Barber ) 48 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Chisholm , Mr. Roderick Robert Crispin 40 Belfast , Ireland , UK Belfast New York City Clark , Mr. Walter Miller 27 Los Angeles , California , US Cherbourg Los Angeles , California , US Clark , Mrs. Virginia Estelle ( née McDowell ) 26 Los Angeles , California , US Cherbourg Los Angeles , California , US Clifford , Mr. George Quincy 40 Stoughton , Massachusetts , US Southampton Stoughton , Massachusetts Colley , Mr. Edward Pomeroy 37 Dublin , Ireland , UK Southampton Vancouver , British Columbia , Canada Compton , Mrs. Mary Eliza ( née Ingersoll ) 64 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Miss Sara Rebecca 39 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Mr. Alexander Taylor Jr . 37 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US Cornell , Mrs. Malvina Helen ( née Lamson ) 55 New York City Southampton New York City Crafton , Mr. John Bertram 59 Roachdale , Indiana , US Southampton Roachdale , Indiana , US Crosby , Captain Edward 70 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 269 Crosby , Mrs. Catherine Elizabeth ( née Halstead ) 64 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Crosby , Miss Harriette Rebecca 39 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Cumings , Mr. John Bradley 39 New York City Cherbourg New York City Cumings , Mrs. Florence Briggs ( née Thayer ) 35 New York City Cherbourg New York , US Daly , Mr. Peter Dennis 51 Lima , Peru Southampton Lima , Peru Daniel , Mr. Robert Williams 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Davidson , Mr. Thornton 31 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Davidson , Mrs. Orian ( née Hays ) 27 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Dick , Mr. Albert Adrian 31 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dick , Mrs. Vera ( née Gillespie ) 17 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dineston , Mrs. Nancy May Bowles 28 London , England , UK Southampton New York City , USA Dineston , George Samuel 30 London , England , UK Southampton New York City , USA Dodge , Dr. Washington 52 San Francisco , California , US Southampton San Francisco , California , US 13 Dodge , Mrs. Ruth ( née Vidaver ) 34 San Francisco , California , US Southampton San Francisco , California , US 5 Dodge , Master Washington , Jr . San Francisco , California , US Southampton San Francisco , California , US 5 Douglas , Mr. Walter Donald 50 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US 62 Douglas , Mrs. Mahala ( née Dutton ) 48 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US and maid , Miss Berthe Leroy 27 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US Douglas , Mrs. Mary Hélène ( née Baxter ) 27 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Duff Gordon , Sir Cosmo Edmund 49 London , England , UK Cherbourg New York City Duff Gordon , Lucy Christiana , Lady ( née Sutherland ) 48 London , England , UK Cherbourg New York City and secretary , Miss Laura Mabel Francatelli 31 London , England , UK Cherbourg New York City Dulles , Mr. William Crothers 39 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 133 Eustis , Miss Elizabeth Mussey 54 Brookline , Massachusetts , US Cherbourg Brookline , Massachusetts , US Evans , Miss Edith Corse 36 New York City Cherbourg New York City Flegenheim , Mrs. Antoinette ( née Wendt ) 48 New York City Cherbourg New York City 7 Flynn , Mr. John Irwin 36 Brooklyn , New York , US Southampton Brooklyn , New York , US 5 Foreman , Mr. Benjamin Laventall 30 New York City Southampton New York City Fortune , Mr. Mark 64 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Fortune , Mrs. Mary ( née McDougald ) 60 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Ethel Flora 28 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Alice Elizabeth 24 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Mabel Helen 23 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Mr. Charles Alexander 19 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Franklin , Mr. Thomas Parnham 37 London , England , UK Southampton New York City Frauenthal , Dr. Henry William 49 London , England , UK Southampton New York City 5 Frauenthal , Mrs. Clara ( née Heinsheimer ) 42 London , England , UK Southampton New York City 5 Frauenthal , Mr. Isaac Gerald 43 London , England , UK Southampton New York City 5 Frölicher , Mr. Maximilian Josef 60 Zürich , Switzerland Cherbourg New York City 5 Frölicher , Mrs. Margaretha Emerentia ( née Stehli ) 48 Zürich , Switzerland Cherbourg New York City 5 Frölicher - Stehli , Miss Hedwig Margaritha 22 Zürich , Switzerland Cherbourg New York City 5 Futrelle , Mr. Jacques Heath 37 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Futrelle , Mrs. Lily May ( née Peel ) 35 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Gee , Mr. Arthur H . 47 St. Annes - on - Sea , Lancashire , England , UK Southampton Mexico City , Mexico 275 Gibson , Mrs. Pauline Caroline ( née Boeson ) 44 New York City Cherbourg New York City 7 Gibson , Miss Dorothy Winifred 22 New York City Cherbourg New York City 7 Goldenberg , Mr. Samuel L . 47 Paris , France Cherbourg New York City 5 Goldenberg , Mrs. Nella ( née Wiggins ) 40 Paris , France Cherbourg New York City 5 Goldschmidt , Mr. George B . 71 New York City Cherbourg New York City Gracie IV , Colonel Archibald 53 Washington , D.C. , US Southampton Washington , D.C. , US Graham , Mr. George Edward 38 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 147 Graham , Mrs. Edith Ware ( née Junkins ) 59 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Graham , Miss Margaret Edith 19 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US and governess , Miss Elizabeth Weed Shutes 40 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Greenfield , Mrs. Blanche ( née Strouse ) 45 New York City Cherbourg New York City 7 Greenfield , Mr. William Bertram 23 New York City Cherbourg New York City 7 Guggenheim , Mr. Benjamin 46 Paris , France Cherbourg New York City and valet , Mr. Victor Giglio 24 Paris , France Cherbourg New York City and chauffeur , Mr. René Pernot 39 Paris , France Cherbourg New York City Harder , Mr. George Achilles 25 New York City Cherbourg New York City 5 Harder , Mrs. Dorothy ( née Annan ) 21 New York City Cherbourg New York City 5 Harper , Mr. Henry Sleeper 48 New York City Cherbourg New York City and dragoman , Mr. Hammad Hassab 27 Cairo , Egypt Cherbourg New York City Harper , Mrs. Myna ( née Haxtun ) 49 New York City Cherbourg New York City Harris , Mr. Henry Birkhardt 45 New York City Southampton New York City Harris , Mrs. Irene ( née Wallach ) 35 New York City Southampton New York City Hawksford , Mr. Walter James 45 Kingston , Surrey , England , UK Southampton New York City Hays , Mr. Charles Melville 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 307 and clerk , Mr. Vivian Ponsonby Payne 22 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Mrs. Clara Jennings ( née Grigg ) 52 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Mary Anne Perreault 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Miss Margaret Bechstein 24 New York City Cherbourg New York City 7 Head , Mr. Christopher 42 Chelsea , London , England , UK Southampton New York City Hilliard , Mr. Herbert Henry 44 Brighton , Massachusetts , US Southampton Brighton , Massachusetts , US Hipkins , Mr. William Edward 55 Birmingham , West Midlands , England , UK Southampton New York City Hippach , Mrs. Ida Sophia ( née Fischer ) 44 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hippach , Miss Jean Gertrude 17 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hogeboom , Mrs. Anna Louisa ( née Andrews ) 51 Hudson , New York , US Cherbourg Hudson , New York , US 10 Holverson , Mr. Alexander Oskar 42 New York City Southampton New York City 38 Holverson , Mrs. Mary Aline ( née Towner ) 35 New York City Southampton New York City 8 Homer , Mr. Harry ( alias E. Haven ) 40 Indianapolis , Indiana , US Southampton Indianapolis , Indiana , US 15 Hoyt , Mr. Frederick Maxfield 35 New York City Southampton Stamford , Connecticut , US Hoyt , Mrs. Jane Anne ( née Forby ) 31 New York City Southampton Stamford , Connecticut , US Hoyt , Mr. William Fisher 42 New York City Cherbourg New York City 14 Isham , Miss Ann Elizabeth 50 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Ismay , Mr. Joseph Bruce 49 Liverpool , Merseyside , England , UK Southampton New York City and valet , Mr. John Richard Fry 39 Liverpool , Merseyside , England , UK Southampton New York City and secretary , Mr. William Henry Harrison 45 Wallasey , Merseyside , England , UK Southampton New York City 110 Johnson , Mrs. Madeline Eliza ( née Austin ) 40 London , England , UK Southampton New York , New York , US 6 Johnson ( later changed to Langston ) , Mr. Samuel James 10 London , England , UK Southampton New York , New York , US Jones , Mr. Charles Cresson 46 Bennington , Vermont , US Southampton Bennington , Vermont , US 80 Julian , Mr. Henry Forbes 50 Torquay , Devon , England , UK Southampton San Francisco , US Kent , Mr. Edward Austin 58 Buffalo , New York , US Cherbourg Buffalo , New York , US 258 Kenyon , Mr. Frederick R . 41 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US Kenyon , Mrs. Marion ( née Stauffer ) 31 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US 8 Kimball , Mr. Edwin Nelson Jr . 42 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Kimball , Mrs. Gertrude ( née Parsons ) 45 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Klaber , Mr. Herman 45 Portland , Oregon , US Southampton Portland , Oregon , US Langston , Ms. Maxine Jane 25 Duncansville , Pennsylvania , US Southampton Duncansivlle , Pennsylvania , US Lambert - Williams , Mr. Fletcher Fellows 43 London , England , UK Southampton Newark , New Jersey , US Leader , Dr. Alice ( née Farnham ) 49 New York City Southampton New York City 8 Leslie , Lucy Noël Martha , Countess of Rothes ( née Dyer - Edwardes ) 27 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 and maid , Miss Roberta Elizabeth Mary `` Cissy '' Maioni 20 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 Lewy , Mr. Ervin G . 31 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Lindeberg - Lind , Mr. Erik Gustav ( alias Edward Lingrey ) 42 Jordanstorp , Södermanland , Sweden Southampton New York City Lindström , Mrs. Sigrid ( née Posse ) 55 Stockholm , Sweden Cherbourg New York City 6 Lines , Mrs. Elizabeth Lindsey ( née James ) 50 Paris , France Cherbourg Hanover , New Hampshire , US 9 Lines , Miss Mary Conover 16 Paris , France Cherbourg Hanover , New Hampshire , US 9 Long , Mr. Milton Clyde 29 Springfield , Massachusetts , US Southampton Springfield , Massachusetts , US 126 Longley , Miss Gretchen Fiske 21 Hudson , New York , US Cherbourg Hudson , New York , US 10 Loring , Mr. Joseph Holland 30 London , England , UK Southampton New York City Madill , Miss Georgette Alexandra 16 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Maguire , Mr. John Edward 30 Brockton , Massachusetts , US Southampton Brockton , Massachusetts , US Maréchal , Mr. Pierre , Sr . 28 Paris , France Cherbourg New York City 7 Marvin , Mr. Daniel Warner 18 New York City Southampton New York City Marvin , Mrs. Mary Graham Carmichael ( née Farquarson ) 18 New York City Southampton New York City 10 Mayné , Miss Bertha Antonine 24 Brussels , Belgium Cherbourg Montreal , Quebec , Canada 6 McCaffry , Mr. Thomas Francis 46 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 292 McCarthy , Mr. Timothy J . 54 Dorchester , Massachusetts , US Southampton Dorchester , Massachusetts , US 175 McGough , Mr. James Robert 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 7 Meyer , Mr. Edgar Joseph 28 New York City Cherbourg New York City Meyer , Mrs. Leila ( née Saks ) 25 New York City Cherbourg New York City 6 Millet , Mr. Francis Davis 65 East Bridgewater , Massachusetts , US Cherbourg East Bridgewater , Massachusetts , US 249 Minahan , Dr. William Edward 44 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 230 Minahan , Mrs. Lillian E. ( née Thorpe ) 37 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Minahan , Miss Daisy E . 33 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Mock , Mr. Philipp Edmund 30 New York City Cherbourg New York City 11 Molson , Mr. Harry Markland 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Moore , Mr. Clarence Bloomfield 47 Washington , D.C. , US Southampton Washington , D.C. , US and valet , Mr. Charles Henry Harrington 37 Washington , D.C. , US Southampton Washington , D.C. , US Natsch , Mr. Charles 36 Brooklyn , New York , US Cherbourg Brooklyn , New York , US Newell , Mr. Arthur Webster 58 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 122 Newell , Miss Madeleine 31 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newell , Miss Marjorie Anne 23 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newsom , Miss Helen Monypeny 19 New York City Southampton New York City 5 Nicholson , Mr. Arthur Ernest 64 Shanklin , Isle of Wight , England , UK Southampton New York City 263 Nourney , Mr. Alfred 20 Cologne , German Empire Cherbourg New York City 7 Omont , Mr. Alfred Fernand 29 Le Havre , France Cherbourg New York City 7 Ostby , Mr. Engelhart Cornelius 64 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 234 Ostby , Miss Helene Ragnhild 22 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 5 Ovies y Rodriguez , Mr. Servando José Florentino 36 Havana , Cuba Cherbourg Havana , Cuba 189 Parr , Mr. William Henry Marsh 29 Belfast , Ireland , UK Belfast New York City Partner , Mr. Austin 40 Tolworth , London , England , UK Southampton Toronto , Ontario , Canada 166 Pears , Mr. Thomas Clinton 29 Isleworth , London , England , UK Southampton New York City Pears , Mrs. Edith ( née Wearne ) 22 Isleworth , London , England , UK Southampton New York City 8 Peñasco y Castellana , Mr. Victor 24 Madrid , Spain Cherbourg New York City Peñasco y Castellana , Mrs. Maria Josefa ( née Perez de Soto y Vallejo ) 22 Madrid , Spain Cherbourg New York City 8 and maid , Doña Fermina Oliva y Ocana 39 Madrid , Spain Cherbourg New York City 8 Peuchen , Major Arthur Godfrey 52 Toronto , Ontario , Canada Southampton Toronto , Ontario , Canada 6 Porter , Mr. Walter Chamberlain 46 Worcester , Massachusetts , US Southampton Worcester , Massachusetts , US 207 Potter , Mrs. Lily Alexenia ( née Wilson ) 56 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 7 Reuchlin , The Honourable Mr. Johan George 38 Rotterdam , The Netherlands Southampton New York City Rheims , Mr. George Alexander Lucien 36 Paris , France Southampton New York City Robert , Mrs. Elisabeth Walton ( née McMillan ) 43 St. Louis , Missouri , US Southampton St. Louis , Missouri , US and maid , Miss Emilie Kreuchen 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Roebling , Mr. Washington Augustus II 31 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Romaine , Mr. Charles Hallance ( alias C. Rolmane ) 45 New York City Southampton New York City 9 Rood , Mr. Hugh Roscoe 38 Seattle , Washington , US Southampton Seattle , Washington , US Rosenbaum , Miss Edith Louise 34 Paris , France Cherbourg Cincinnati , Ohio , US 11 Rosenshine , Mr. George ( alias George Thorne ) 46 New York City Cherbourg New York City 16 Ross , Mr. John Hugo 36 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Rothschild , Mr. Martin 46 New York City Cherbourg New York City Rothschild , Mrs. Elizabeth Jane Anne ( née Barrett ) 54 New York City Cherbourg New York City 6 Rowe , Mr. Alfred G . 59 Liverpool , Merseyside , England , UK Southampton New York City 109 Ryerson , Mr. Arthur Larned 61 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Mrs. Emily Maria ( née Borie ) 48 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and maid , Miss Victorine Chaudanson 36 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Susan Parker `` Suzette '' 21 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Emily Borie 18 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Master John Borie `` Jack '' 13 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and governess , Miss Grace Scott Bowen 45 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Saalfeld , Mr. Adolphe 47 Manchester , England , UK Southampton New York City Salomon , Mr. Abraham Lincoln 43 New York City Cherbourg New York City Schabert , Mrs. Emma ( née Mock ) 35 Hamburg , German Empire Southampton New York City 11 Seward , Mr. Frederic Kimber 34 New York City Southampton New York City 7 Silverthorne , Mr. Spencer Victor 35 St. Louis , Missouri , US Southampton St. Louis , Missouri , US 5 Silvey , Mr. William Baird 50 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US Silvey , Mrs. Alice ( née Munger ) 39 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US 11 Simonius - Blumer , Colonel Alfons 56 Basel , Switzerland Southampton New York City Sloper , Mr. William Thompson 28 New Britain , Connecticut , US Southampton New Britain , Connecticut , US 7 Smart , Mr. John Montgomery 56 Kildale , North Yorkshire , England , UK Southampton New York City Smith , Mr. James Clinch 56 Paris , France Cherbourg Long Island , New York , US Smith , Mr. Lucien Philip 24 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US Smith , Mrs. Mary Eloise ( née Hughes ) 18 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US 6 Smith , Mr. Richard William 57 Streatham , London , England , UK Southampton New York City Snyder , Mr. John Pillsbury 24 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Snyder , Mrs. Nellie ( née Stevenson ) 23 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Spedden , Mr. Frederic Oakley 45 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Mrs. Margaretta Corning ( née Stone ) 39 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and maid , Miss Helen Alice Wilson 31 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Master Robert Douglas 6 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and nurse , Miss Elizabeth Margaret Burns 41 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spencer , Mr. William Augustus 57 New York City Cherbourg New York City Spencer , Mrs. Marie Eugénie ( née Demougeot ) 45 New York City Cherbourg New York City 6 and maid , Miss Eugenie Elise Lurette 59 New York City Cherbourg New York City 6 Stähelin - Maeglin , Dr. Max 32 Basel , Switzerland Southampton New York City Stead , Mr. William Thomas 62 London , England , UK Southampton New York City Stengel , Mr. Charles Emil Henry 54 Newark , New Jersey , US Cherbourg Newark , New Jersey , US Stengel , Mrs. Annie May ( née Morris ) 44 Newark , New Jersey , US Cherbourg Newark , New Jersey , US 5 Stephenson , Mrs. Martha ( née Eustis ) 52 Paris , France Cherbourg Haverford , Pennsylvania , US Stewart , Mr. Albert A . 54 Cincinnati , Ohio , US Cherbourg Gallipolis , Ohio , US Stone , Mrs. Martha Evelyn ( née Stevens ) 62 New York City Southampton New York City 6 and maid , Miss Amelie `` Amelia '' Icard 38 New York City Southampton New York City 6 Straus , Mr. Isidor 67 New York City Southampton New York City 96 and valet , Mr. John Farthing 57 New York City Southampton New York City Straus , Mrs. Rosalie Ida ( née Blun ) 63 New York City Southampton New York City and maid , Miss Ellen Bird 31 New York City Southampton New York City 8 Sutton , Mr. Frederick 61 Haddonfield , New Jersey , US Southampton Haddonfield , New Jersey , US 46 Swift , Mrs. Margaret Welles ( née Barron ) 46 New York City Southampton New York City 8 Taussig , Mr. Emil 52 New York City Southampton New York City Taussig , Mrs. Tillie ( née Mandelbaum ) 39 New York City Southampton New York City 8 Taussig , Miss Ruth 18 New York City Southampton New York City 8 Taylor , Mr. Elmer Zebley 48 London , England , UK Southampton East Orange , New Jersey , US 5 Taylor , Mrs. Juliet Cummins ( née Wright ) 48 London , England , UK Southampton East Orange , New Jersey , US 5 Thayer , Mr. John Borland II 49 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mrs. Marian Longsteth ( née Morris ) 39 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US and maid , Miss Margaret Fleming 42 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mr. John Borland `` Jack '' III 17 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thorne , Mrs. Gertrude Maybelle ( née McMinn ) 37 New York City , New York , US Cherbourg New York City , New York , US Tucker , Mr. Gilbert Milligan Jr . 31 Albany , New York , US Cherbourg Albany , New York , US 7 Uruchurtu , Don Manuel E . 40 Mexico City , Mexico Cherbourg Mexico City , Mexico Van der Hoef , Mr. Wyckoff 61 Brooklyn , New York , US Belfast Brooklyn , New York , US 245 Walker , Mr. William Anderson 48 East Orange , New Jersey , US Southampton East Orange , New Jersey , US Warren , Mr. Frank Manley 63 Portland , Oregon , US Cherbourg Portland , Oregon , US Warren , Mrs. Anna Sophia ( née Atkinson ) 60 Portland , Oregon , US Cherbourg Portland , Oregon , US 5 Weir , Colonel John 59 New York City Southampton New York City White , Mr. Percival Wayland 54 Brunswick , Maine , US Southampton Brunswick , Maine , US White , Mr. Richard Frasar 21 Brunswick , Maine , US Southampton Brunswick , Maine , US 169 White , Mrs. Ella ( née Holmes ) 55 New York City Cherbourg New York City 8 and maid , Miss Amelia Mayo `` Nellie '' Bessette 39 New York City Cherbourg New York City 8 and manservant , Mr. Sante Righini 22 New York City Cherbourg New York City 232 Wick , Colonel George Dennick 58 Youngstown , Ohio , US Southampton Youngstown , Ohio , US Wick , Mrs. Mary ( née Hitchcock ) 45 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Wick , Miss Mary Natalie 31 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Widener , Mr. George Dunton 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and valet , Mr. Edwin Herbert Keeping 33 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 45 Widener , Mrs. Eleanor ( née Elkins ) 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and maid , Miss Amalie Henriette `` Emily '' Gieger 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Widener , Mr. Harry Elkins 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Willard , Miss Constance 21 Duluth , Minnesota , US Southampton Duluth , Minnesota , US 8 Williams , Mr. Charles Duane 51 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Williams , Mr. Richard Norris II 21 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Woolner , Mr. Hugh 45 London , England , UK Southampton New York City Wright , Mr. George 62 Halifax , Nova Scotia , Canada Southampton Halifax , Nova Scotia , Canada Young , Miss Marie Grice 36 New York City Cherbourg New York City 8
  Fact 0.0: Allen, Miss Elizabeth Walton was 29 years old.
  Fact 0.1: Allen, Miss Elizabeth Walton is from St. Louis, Missouri, US.
  Fact 0.2: Allen, Miss Elizabeth Walton boarded at Southampton.
  Fact 0.3: Allen, Miss Elizabeth Walton's destination is St. Louis, Missouri, US.
  Fact 0.4: Allison, Mr. Hudson Joshua Creighton was 30 years old.
  Fact 0.5: Allison, Mr. Hudson Joshua Creighton is from Montreal, Quebec, Canada.
  Fact 0.6: Allison, Mr. Hudson Joshua Creighton boarded at Southampton.
  Fact 0.7: Allison, Mr. Hudson Joshua Creighton's destination is Montreal, Quebec, Canada.
  Fact 0.8: Allison, Mr. Hudson Joshua Creighton has 135 in the lifeboat or body column.
  Fact 0.9: Mr. George Swane was 19 years old.
  Fact 0.10: Mr. George Swane is listed as the chauffeur.
  Fact 0.11: Mr. George Swane is from Montreal, Quebec, Canada.
  Fact 0.12: Mr. George Swane boarded at Southampton.
  Fact 0.13: Mr. George Swane's destination is Montreal, Quebec, Canada.
  Fact 0.14: Mr. George Swane has 294 in the lifeboat or body column.
  Fact 0.15: Miss Amelia Mary "Mildred" Brown was 18 years old.
  Fact 0.16: Miss Amelia Mary "Mildred" Brown is listed as the cook.
  Fact 0.17: Miss Amelia Mary "Mildred" Brown is from London, England, UK.
  Fact 0.18: Miss Amelia Mary "Mildred" Brown boarded at Southampton.
  Fact 0.19: Miss Amelia Mary "Mildred" Brown's destination is Montreal, Quebec, Canada.
  Fact 0.20: Miss Amelia Mary "Mildred" Brown has 11 in the lifeboat or body column.
  Fact 0.21: Allison, Mrs. Bessie Waldo (née Daniels) was 25 years old.
  Fact 0.22: Allison, Mrs. Bessie Waldo (née Daniels) is from Montreal, Quebec, Canada.
  Fact 0.23: Allison, Mrs. Bessie Waldo (née Daniels) boarded at Southampton.
  Fact 0.24: Allison, Mrs. Bessie Waldo (née Daniels)'s destination is Montreal, Quebec, Canada.
  Fact 0.25: Miss Sarah Daniels was 33 years old.
  Fact 0.26: Miss Sarah Daniels is listed as the maid.
  Fact 0.27: Miss Sarah Daniels is from Montreal, Quebec, Canada.
  Fact 0.28: Miss Sarah Daniels boarded at Southampton.
  Fact 0.29: Miss Sarah Daniels's destination is Montreal, Quebec, Canada.
  Fact 0.30: Miss Sarah Daniels has 8 in the lifeboat or body column.
  Fact 0.31: Allison, Miss Helen Loraine is from Montreal, Quebec, Canada.
  Fact 0.32: Allison, Miss Helen Loraine boarded at Southampton.
  Fact 0.33: Allison, Miss Helen Loraine's destination is Montreal, Quebec, Canada.
  Fact 0.34: Allison, Master Hudson Trevor is listed as age 0.11 and as 11 months.
  Fact 0.35: Allison, Master Hudson Trevor is from Montreal, Quebec, Canada.
  Fact 0.36: Allison, Master Hudson Trevor boarded at Southampton.
  Fact 0.37: Allison, Master Hudson Trevor's destination is Montreal, Quebec, Canada.
  Fact 0.38: Allison, Master Hudson Trevor has 11 in the lifeboat or body column.
  Fact 0.39: Miss Alice Catherine Cleaver was 22 years old.
  Fact 0.40: Miss Alice Catherine Cleaver is listed as the nurse.
  Fact 0.41: Miss Alice Catherine Cleaver is from London, England, UK.
  Fact 0.42: Miss Alice Catherine Cleaver boarded at Southampton.
  Fact 0.43: Miss Alice Catherine Cleaver's destination is Montreal, Quebec, Canada.
  Fact 0.44: Miss Alice Catherine Cleaver has 11 in the lifeboat or body column.
  Fact 0.45: Anderson, Mr. Harry was 47 years old.
  Fact 0.46: Anderson, Mr. Harry is from New York City, New York, US.
  Fact 0.47: Anderson, Mr. Harry boarded at Southampton.
  Fact 0.48: Anderson, Mr. Harry's destination is New York City, New York, US.
  Fact 0.49: Andrews, Miss Kornelia Theodosia was 62 years old.
  Fact 0.50: Andrews, Miss Kornelia Theodosia is from Hudson, New York, US.
  Fact 0.51: Andrews, Miss Kornelia Theodosia boarded at Cherbourg.
  Fact 0.52: Andrews, Miss Kornelia Theodosia's destination is Hudson, New York, US.
  Fact 0.53: Andrews, Miss Kornelia Theodosia has 10 in the lifeboat or body column.
  Fact 0.54: Andrews, Mr. Thomas was 39 years old.
  Fact 0.55: Andrews, Mr. Thomas is from Belfast, Ireland, UK.
  Fact 0.56: Andrews, Mr. Thomas boarded at Belfast.
  Fact 0.57: Andrews, Mr. Thomas's destination is New York City.
  Fact 0.58: Appleton, Mrs. Charlotte Lane (née Lamson) was 53 years old.
  Fact 0.59: Appleton, Mrs. Charlotte Lane (née Lamson) is from New York City.
  Fact 0.60: Appleton, Mrs. Charlotte Lane (née Lamson) boarded at Southampton.
  Fact 0.61: Appleton, Mrs. Charlotte Lane (née Lamson)'s destination is New York City.
  Fact 0.62: Artagaveytia, Mr. Ramon was 71 years old.
  Fact 0.63: Artagaveytia, Mr. Ramon is from Buenos Aires, Argentina.
  Fact 0.64: Artagaveytia, Mr. Ramon boarded at Cherbourg.
  Fact 0.65: Artagaveytia, Mr. Ramon's destination is New York City.
  Fact 0.66: Artagaveytia, Mr. Ramon has 22 in the lifeboat or body column.
  Fact 0.67: Astor, Colonel John Jacob IV was 47 years old.
  Fact 0.68: Astor, Colonel John Jacob IV is from New York City.
  Fact 0.69: Astor, Colonel John Jacob IV boarded at Cherbourg.
  Fact 0.70: Astor, Colonel John Jacob IV's destination is New York City.
  Fact 0.71: Astor, Colonel John Jacob IV has 124 in the lifeboat or body column.
  Fact 0.72: Astor, Mrs. Madeleine Talmage (née Force) was 18 years old.
  Fact 0.73: Astor, Mrs. Madeleine Talmage (née Force) is from New York City.
  Fact 0.74: Astor, Mrs. Madeleine Talmage (née Force) boarded at Cherbourg.
  Fact 0.75: Astor, Mrs. Madeleine Talmage (née Force)'s destination is New York City.
  Fact 0.76: Miss Rosalie Bidois was 46 years old.
  Fact 0.77: Miss Rosalie Bidois is listed as the maid.
  Fact 0.78: Miss Rosalie Bidois is from New York City.
  Fact 0.79: Miss Rosalie Bidois boarded at Cherbourg.
  Fact 0.80: Miss Rosalie Bidois's destination is New York City.
  Fact 0.81: Miss Caroline Louise Endres was 39 years old.
  Fact 0.82: Miss Caroline Louise Endres is listed as the nurse.
  Fact 0.83: Miss Caroline Louise Endres is from New York City.
  Fact 0.84: Miss Caroline Louise Endres boarded at Cherbourg.
  Fact 0.85: Miss Caroline Louise Endres's destination is New York City.
  Fact 0.86: Aubart, Mrs. Léontine Pauline was 24 years old.
  Fact 0.87: Aubart, Mrs. Léontine Pauline is from Paris, France.
  Fact 0.88: Aubart, Mrs. Léontine Pauline boarded at Cherbourg.
  Fact 0.89: Aubart, Mrs. Léontine Pauline's destination is New York City.
  Fact 0.90: Aubart, Mrs. Léontine Pauline has 9 in the lifeboat or body column.
  Fact 0.91: Miss Emma Sägesser was 24 years old.
  Fact 0.92: Miss Emma Sägesser is listed as the maid.
  Fact 0.93: Miss Emma Sägesser is from Paris, France.
  Fact 0.94: Miss Emma Sägesser boarded at Cherbourg.
  Fact 0.95: Miss Emma Sägesser's destination is New York City.
  Fact 0.96: Miss Emma Sägesser has 9 in the lifeboat or body column.
  Fact 0.97: Barkworth, Mr. Algernon Henry Wilson was 47 years old.
  Fact 0.98: Barkworth, Mr. Algernon Henry Wilson is from Hessle, East Yorkshire, England, UK.
  Fact 0.99: Barkworth, Mr. Algernon Henry Wilson boarded at Southampton.
  Fact 0.100: Barkworth, Mr. Algernon Henry Wilson's destination is New York City.
  Fact 0.101: Baumann, Mr. John D. was 60 years old.
  Fact 0.102: Baumann, Mr. John D. is from New York City.
  Fact 0.103: Baumann, Mr. John D. boarded at Cherbourg.
  Fact 0.104: Baumann, Mr. John D.'s destination is New York City.
  Fact 0.105: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) was 50 years old.
  Fact 0.106: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) is from Montreal, Quebec, Canada.
  Fact 0.107: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) boarded at Cherbourg.
  Fact 0.108: Baxter, Mrs. Hélène (née de Lanaudière-Chaput)'s destination is Montreal, Quebec, Canada.
  Fact 0.109: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) has 6 in the lifeboat or body column.
  Fact 0.110: Baxter, Mr. Quigg Edmond was 24 years old.
  Fact 0.111: Baxter, Mr. Quigg Edmond is from Montreal, Quebec, Canada.
  Fact 0.112: Baxter, Mr. Quigg Edmond boarded at Cherbourg.
  Fact 0.113: Baxter, Mr. Quigg Edmond's destination is Montreal, Quebec, Canada.
  Fact 0.114: Beattie, Mr. Thomson was 36 years old.
  Fact 0.115: Beattie, Mr. Thomson is from Fergus, Ontario, Canada.
  Fact 0.116: Beattie, Mr. Thomson boarded at Southampton.
  Fact 0.117: Beattie, Mr. Thomson's destination is Fergus, Ontario, Canada.
  Fact 0.118: Beattie, Mr. Thomson has 331 in the lifeboat or body column.
  Fact 0.119: Beckwith, Mr. Richard Leonard was 37 years old.
  Fact 0.120: Beckwith, Mr. Richard Leonard is from New York City.
  Fact 0.121: Beckwith, Mr. Richard Leonard boarded at Southampton.
  Fact 0.122: Beckwith, Mr. Richard Leonard's destination is New York City.
  Fact 0.123: Beckwith, Mr. Richard Leonard has 5 in the lifeboat or body column.
  Fact 0.124: Beckwith, Mrs. Sallie (née Monypeny) was 46 years old.
  Fact 0.125: Beckwith, Mrs. Sallie (née Monypeny) is from New York City.
  Fact 0.126: Beckwith, Mrs. Sallie (née Monypeny) boarded at Southampton.
  Fact 0.127: Beckwith, Mrs. Sallie (née Monypeny)'s destination is New York City.
  Fact 0.128: Beckwith, Mrs. Sallie (née Monypeny) has 5 in the lifeboat or body column.
  Fact 0.129: Behr, Mr. Karl Howell was 26 years old.
  Fact 0.130: Behr, Mr. Karl Howell is from New York City.
  Fact 0.131: Behr, Mr. Karl Howell boarded at Cherbourg.
  Fact 0.132: Behr, Mr. Karl Howell's destination is New York City.
  Fact 0.133: Behr, Mr. Karl Howell has 5 in the lifeboat or body column.
  Fact 0.134: Birnbaum, Mr. Jakob was 24 years old.
  Fact 0.135: Birnbaum, Mr. Jakob is from Antwerp, Belgium.
  Fact 0.136: Birnbaum, Mr. Jakob boarded at Cherbourg.
  Fact 0.137: Birnbaum, Mr. Jakob's destination is San Francisco, California, US.
  Fact 0.138: Birnbaum, Mr. Jakob has 148 in the lifeboat or body column.
  Fact 0.139: Bishop, Mr. Dickinson H. "Dick" was 25 years old.
  Fact 0.140: Bishop, Mr. Dickinson H. "Dick" is from Dowagiac, Michigan, US.
  Fact 0.141: Bishop, Mr. Dickinson H. "Dick" boarded at Cherbourg.
  Fact 0.142: Bishop, Mr. Dickinson H. "Dick"'s destination is Dowagiac, Michigan, US.
  Fact 0.143: Bishop, Mr. Dickinson H. "Dick" has 7 in the lifeboat or body column.
  Fact 0.144: Bishop, Mrs. Helen (née Walton) was 19 years old.
  Fact 0.145: Bishop, Mrs. Helen (née Walton) is from Dowagiac, Michigan, US.
  Fact 0.146: Bishop, Mrs. Helen (née Walton) boarded at Cherbourg.
  Fact 0.147: Bishop, Mrs. Helen (née Walton)'s destination is Dowagiac, Michigan, US.
  Fact 0.148: Bishop, Mrs. Helen (née Walton) has 7 in the lifeboat or body column.
  Fact 0.149: Björnström-Steffanson, Mr. Mauritz Håkan was 28 years old.
  Fact 0.150: Björnström-Steffanson, Mr. Mauritz Håkan is from Stockholm, Sweden.
  Fact 0.151: Björnström-Steffanson, Mr. Mauritz Håkan boarded at Southampton.
  Fact 0.152: Björnström-Steffanson, Mr. Mauritz Håkan's destination is Washington, D.C., US.
  Fact 0.153: Blackwell, Mr. Stephen Weart was 45 years old.
  Fact 0.154: Blackwell, Mr. Stephen Weart is from Trenton, New Jersey, US.
  Fact 0.155: Blackwell, Mr. Stephen Weart boarded at Southampton.
  Fact 0.156: Blackwell, Mr. Stephen Weart's destination is Trenton, New Jersey, US.
  Fact 0.157: Blank, Mr. Henry was 39 years old.
  Fact 0.158: Blank, Mr. Henry is from Glen Ridge, New Jersey, US.
  Fact 0.159: Blank, Mr. Henry boarded at Cherbourg.
  Fact 0.160: Blank, Mr. Henry's destination is Glen Ridge, New Jersey, US.
  Fact 0.161: Blank, Mr. Henry has 7 in the lifeboat or body column.
  Fact 0.162: Bonnell, Miss Elizabeth was 61 years old.
  Fact 0.163: Bonnell, Miss Elizabeth is from Youngstown, Ohio, US.
  Fact 0.164: Bonnell, Miss Elizabeth boarded at Southampton.
  Fact 0.165: Bonnell, Miss Elizabeth's destination is Youngstown, Ohio, US.
  Fact 0.166: Bonnell, Miss Elizabeth has 8 in the lifeboat or body column.
  Fact 0.167: Bonnell, Miss Caroline was 30 years old.
  Fact 0.168: Bonnell, Miss Caroline is from Youngstown, Ohio, US.
  Fact 0.169: Bonnell, Miss Caroline boarded at Southampton.
  Fact 0.170: Bonnell, Miss Caroline's destination is Youngstown, Ohio, US.
  Fact 0.171: Bonnell, Miss Caroline has 8 in the lifeboat or body column.
  Fact 0.172: Borebank, Mr. John James was 42 years old.
  Fact 0.173: Borebank, Mr. John James is from London, England, UK.
  Fact 0.174: Borebank, Mr. John James boarded at Southampton.
  Fact 0.175: Borebank, Mr. John James's destination is Toronto, Ontario, Canada.
  Fact 0.176: Bowerman, Miss Elsie Edith was 22 years old.
  Fact 0.177: Bowerman, Miss Elsie Edith is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.178: Bowerman, Miss Elsie Edith boarded at Southampton.
  Fact 0.179: Bowerman, Miss Elsie Edith's destination is New York City.
  Fact 0.180: Bowerman, Miss Elsie Edith has 6 in the lifeboat or body column.
  Fact 0.181: Brady, Mr. John Bertram was 41 years old.
  Fact 0.182: Brady, Mr. John Bertram is from Pomeroy, Washington, US.
  Fact 0.183: Brady, Mr. John Bertram boarded at Southampton.
  Fact 0.184: Brady, Mr. John Bertram's destination is Pomeroy, Washington, US.
  Fact 0.185: Brandeis, Mr. Emil was 48 years old.
  Fact 0.186: Brandeis, Mr. Emil is from Omaha, Nebraska, US.
  Fact 0.187: Brandeis, Mr. Emil boarded at Cherbourg.
  Fact 0.188: Brandeis, Mr. Emil's destination is Omaha, Nebraska, US.
  Fact 0.189: Brandeis, Mr. Emil has 208 in the lifeboat or body column.
  Fact 0.190: Brereton, Mr. George Andrew (alias George A. Brayton) was 37 years old.
  Fact 0.191: Brereton, Mr. George Andrew (alias George A. Brayton) is from Los Angeles, US.
  Fact 0.192: Brereton, Mr. George Andrew (alias George A. Brayton) boarded at Southampton.
  Fact 0.193: Brereton, Mr. George Andrew (alias George A. Brayton)'s destination is Los Angeles, California, US.
  Fact 0.194: Brereton, Mr. George Andrew (alias George A. Brayton) has 9 in the lifeboat or body column.
  Fact 0.195: Brewe, Dr. Arthur Jackson was 45 years old.
  Fact 0.196: Brewe, Dr. Arthur Jackson is from Philadelphia, Pennsylvania, US.
  Fact 0.197: Brewe, Dr. Arthur Jackson boarded at Cherbourg.
  Fact 0.198: Brewe, Dr. Arthur Jackson's destination is Philadelphia, Pennsylvania, US.
  Fact 0.199: Brown, Mrs. Caroline Lane (née Lamson) was 59 years old.
  Fact 0.200: Brown, Mrs. Caroline Lane (née Lamson) is from Belmont, Massachusetts, US.
  Fact 0.201: Brown, Mrs. Caroline Lane (née Lamson) boarded at Southampton.
  Fact 0.202: Brown, Mrs. Caroline Lane (née Lamson)'s destination is Belmont, Massachusetts, US.
  Fact 0.203: Brown, Mrs. Margaret (née Tobin) was 44 years old.
  Fact 0.204: Brown, Mrs. Margaret (née Tobin) is from Denver, Colorado, US.
  Fact 0.205: Brown, Mrs. Margaret (née Tobin) boarded at Cherbourg.
  Fact 0.206: Brown, Mrs. Margaret (née Tobin)'s destination is Denver, Colorado, US.
  Fact 0.207: Brown, Mrs. Margaret (née Tobin) has 6 in the lifeboat or body column.
  Fact 0.208: Bucknell, Mrs. Emma Eliza (née Ward) was 59 years old.
  Fact 0.209: Bucknell, Mrs. Emma Eliza (née Ward) is from Philadelphia, Pennsylvania, US.
  Fact 0.210: Bucknell, Mrs. Emma Eliza (née Ward) boarded at Cherbourg.
  Fact 0.211: Bucknell, Mrs. Emma Eliza (née Ward)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.212: Bucknell, Mrs. Emma Eliza (née Ward) has 8 in the lifeboat or body column.
  Fact 0.213: Miss Albina Bazzani was 36 years old.
  Fact 0.214: Miss Albina Bazzani is listed as the maid.
  Fact 0.215: Miss Albina Bazzani is from Philadelphia, Pennsylvania, US.
  Fact 0.216: Miss Albina Bazzani boarded at Cherbourg.
  Fact 0.217: Miss Albina Bazzani's destination is Philadelphia, Pennsylvania, US.
  Fact 0.218: Miss Albina Bazzani has 8 in the lifeboat or body column.
  Fact 0.219: Butt, Major Archibald Willingham was 46 years old.
  Fact 0.220: Butt, Major Archibald Willingham is from Washington, D.C., US.
  Fact 0.221: Butt, Major Archibald Willingham boarded at Southampton.
  Fact 0.222: Butt, Major Archibald Willingham's destination is Washington, D.C., US.
  Fact 0.223: Calderhead, Mr. Edward Pennington was 42 years old.
  Fact 0.224: Calderhead, Mr. Edward Pennington is from New York City.
  Fact 0.225: Calderhead, Mr. Edward Pennington boarded at Southampton.
  Fact 0.226: Calderhead, Mr. Edward Pennington's destination is New York City.
  Fact 0.227: Calderhead, Mr. Edward Pennington has 5 in the lifeboat or body column.
  Fact 0.228: Candee, Mrs. Helen Churchill (née Hungerford) was 53 years old.
  Fact 0.229: Candee, Mrs. Helen Churchill (née Hungerford) is from Washington, D.C., US.
  Fact 0.230: Candee, Mrs. Helen Churchill (née Hungerford) boarded at Cherbourg.
  Fact 0.231: Candee, Mrs. Helen Churchill (née Hungerford)'s destination is Washington, D.C., US.
  Fact 0.232: Candee, Mrs. Helen Churchill (née Hungerford) has 6 in the lifeboat or body column.
  Fact 0.233: Cardeza, Mrs. Charlotte Wardle (née Drake) was 58 years old.
  Fact 0.234: Cardeza, Mrs. Charlotte Wardle (née Drake) is from Germantown, Pennsylvania, US.
  Fact 0.235: Cardeza, Mrs. Charlotte Wardle (née Drake) boarded at Cherbourg.
  Fact 0.236: Cardeza, Mrs. Charlotte Wardle (née Drake)'s destination is Germantown, Pennsylvania, US.
  Fact 0.237: Miss Annie Moore Ward was 38 years old.
  Fact 0.238: Miss Annie Moore Ward is listed as the maid.
  Fact 0.239: Miss Annie Moore Ward is from Germantown, Pennsylvania, US.
  Fact 0.240: Miss Annie Moore Ward boarded at Cherbourg.
  Fact 0.241: Miss Annie Moore Ward's destination is Germantown, Pennsylvania, US.
  Fact 0.242: Cardeza, Mr. Thomas Drake Martinez was 36 years old.
  Fact 0.243: Cardeza, Mr. Thomas Drake Martinez is from Germantown, Pennsylvania, US.
  Fact 0.244: Cardeza, Mr. Thomas Drake Martinez boarded at Cherbourg.
  Fact 0.245: Cardeza, Mr. Thomas Drake Martinez's destination is Germantown, Pennsylvania, US.
  Fact 0.246: Mr. Gustave J. Lesueur was 35 years old.
  Fact 0.247: Mr. Gustave J. Lesueur is listed as the valet.
  Fact 0.248: Mr. Gustave J. Lesueur is from Germantown, Pennsylvania, US.
  Fact 0.249: Mr. Gustave J. Lesueur boarded at Cherbourg.
  Fact 0.250: Mr. Gustave J. Lesueur's destination is Germantown, Pennsylvania, US.
  Fact 0.251: Carlsson, Mr. Frans Olof was 33 years old.
  Fact 0.252: Carlsson, Mr. Frans Olof is from New York City.
  Fact 0.253: Carlsson, Mr. Frans Olof boarded at Southampton.
  Fact 0.254: Carlsson, Mr. Frans Olof's destination is New York City.
  Fact 0.255: Carrau, Mr. Francisco M. was 31 years old.
  Fact 0.256: Carrau, Mr. Francisco M. is from Montevideo, Uruguay.
  Fact 0.257: Carrau, Mr. Francisco M. boarded at Southampton.
  Fact 0.258: Carrau, Mr. Francisco M.'s destination is Montevideo, Uruguay.
  Fact 0.259: Carrau, Mr. José Pedro was 17 years old.
  Fact 0.260: Carrau, Mr. José Pedro is from Montevideo, Uruguay.
  Fact 0.261: Carrau, Mr. José Pedro boarded at Southampton.
  Fact 0.262: Carrau, Mr. José Pedro's destination is Montevideo, Uruguay.
  Fact 0.263: Carter, Mr. William Ernest was 36 years old.
  Fact 0.264: Carter, Mr. William Ernest is from Bryn Mawr, Pennsylvania, US.
  Fact 0.265: Carter, Mr. William Ernest boarded at Southampton.
  Fact 0.266: Carter, Mr. William Ernest's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.267: Mr. Alexander Cairns was 28 years old.
  Fact 0.268: Mr. Alexander Cairns is listed as the valet.
  Fact 0.269: Mr. Alexander Cairns is from Bryn Mawr, Pennsylvania, US.
  Fact 0.270: Mr. Alexander Cairns boarded at Southampton.
  Fact 0.271: Mr. Alexander Cairns's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.272: Mr. Charles Augustus Aldworth was 30 years old.
  Fact 0.273: Mr. Charles Augustus Aldworth is listed as the chauffeur.
  Fact 0.274: Mr. Charles Augustus Aldworth is from Bryn Mawr, Pennsylvania, US.
  Fact 0.275: Mr. Charles Augustus Aldworth boarded at Southampton.
  Fact 0.276: Mr. Charles Augustus Aldworth's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.277: Carter, Mrs. Lucile (née Polk) was 36 years old.
  Fact 0.278: Carter, Mrs. Lucile (née Polk) is from Bryn Mawr, Pennsylvania, US.
  Fact 0.279: Carter, Mrs. Lucile (née Polk) boarded at Southampton.
  Fact 0.280: Carter, Mrs. Lucile (née Polk)'s destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.281: Miss Auguste Serreplan was 30 years old.
  Fact 0.282: Miss Auguste Serreplan is listed as the maid.
  Fact 0.283: Miss Auguste Serreplan is from Bryn Mawr, Pennsylvania, US.
  Fact 0.284: Miss Auguste Serreplan boarded at Southampton.
  Fact 0.285: Miss Auguste Serreplan's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.286: Carter, Miss Lucile Polk was 13 years old.
  Fact 0.287: Carter, Miss Lucile Polk is from Bryn Mawr, Pennsylvania, US.
  Fact 0.288: Carter, Miss Lucile Polk boarded at Southampton.
  Fact 0.289: Carter, Miss Lucile Polk's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.290: Carter, Master William Thornton II was 11 years old.
  Fact 0.291: Carter, Master William Thornton II is from Bryn Mawr, Pennsylvania, US.
  Fact 0.292: Carter, Master William Thornton II boarded at Southampton.
  Fact 0.293: Carter, Master William Thornton II's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.294: Case, Mr. Howard Brown was 49 years old.
  Fact 0.295: Case, Mr. Howard Brown is from Ascot, Berkshire, England, UK.
  Fact 0.296: Case, Mr. Howard Brown boarded at Southampton.
  Fact 0.297: Case, Mr. Howard Brown's destination is Rochester, New York, US.
  Fact 0.298: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) was 36 years old.
  Fact 0.299: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) is from New York City.
  Fact 0.300: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) boarded at Cherbourg.
  Fact 0.301: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick)'s destination is New York City.
  Fact 0.302: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) has 5 in the lifeboat or body column.
  Fact 0.303: Cavendish, Mr. Tyrell William was 36 years old.
  Fact 0.304: Cavendish, Mr. Tyrell William is from London, England, UK.
  Fact 0.305: Cavendish, Mr. Tyrell William boarded at Southampton.
  Fact 0.306: Cavendish, Mr. Tyrell William's destination is New York City.
  Fact 0.307: Cavendish, Mr. Tyrell William has 172 in the lifeboat or body column.
  Fact 0.308: Cavendish, Mrs. Julia Florence (née Siegel) was 25 years old.
  Fact 0.309: Cavendish, Mrs. Julia Florence (née Siegel) is from London, England, UK.
  Fact 0.310: Cavendish, Mrs. Julia Florence (née Siegel) boarded at Southampton.
  Fact 0.311: Cavendish, Mrs. Julia Florence (née Siegel)'s destination is New York City.
  Fact 0.312: Cavendish, Mrs. Julia Florence (née Siegel) has 6 in the lifeboat or body column.
  Fact 0.313: Miss Ellen "Nellie" Barber was 26 years old.
  Fact 0.314: Miss Ellen "Nellie" Barber is listed as the maid.
  Fact 0.315: Miss Ellen "Nellie" Barber is from London, England, UK.
  Fact 0.316: Miss Ellen "Nellie" Barber boarded at Southampton.
  Fact 0.317: Miss Ellen "Nellie" Barber's destination is New York City.
  Fact 0.318: Miss Ellen "Nellie" Barber has 6 in the lifeboat or body column.
  Fact 0.319: Chaffee, Mr. Hubert Fuller was 46 years old.
  Fact 0.320: Chaffee, Mr. Hubert Fuller is from Amenia, North Dakota, US.
  Fact 0.321: Chaffee, Mr. Hubert Fuller boarded at Southampton.
  Fact 0.322: Chaffee, Mr. Hubert Fuller's destination is Amenia, North Dakota, US.
  Fact 0.323: Chaffee, Mrs. Carrie Constance (née Toogood) was 47 years old.
  Fact 0.324: Chaffee, Mrs. Carrie Constance (née Toogood) is from Amenia, North Dakota, US.
  Fact 0.325: Chaffee, Mrs. Carrie Constance (née Toogood) boarded at Southampton.
  Fact 0.326: Chaffee, Mrs. Carrie Constance (née Toogood)'s destination is Amenia, North Dakota, US.
  Fact 0.327: Chambers, Mr. Norman Campbell was 27 years old.
  Fact 0.328: Chambers, Mr. Norman Campbell is from New York City.
  Fact 0.329: Chambers, Mr. Norman Campbell boarded at Southampton.
  Fact 0.330: Chambers, Mr. Norman Campbell's destination is New York City.
  Fact 0.331: Chambers, Mr. Norman Campbell has 5 in the lifeboat or body column.
  Fact 0.332: Chambers, Mrs. Bertha (née Griggs) was 32 years old.
  Fact 0.333: Chambers, Mrs. Bertha (née Griggs) is from New York City.
  Fact 0.334: Chambers, Mrs. Bertha (née Griggs) boarded at Southampton.
  Fact 0.335: Chambers, Mrs. Bertha (née Griggs)'s destination is New York City.
  Fact 0.336: Chambers, Mrs. Bertha (née Griggs) has 5 in the lifeboat or body column.
  Fact 0.337: Cherry, Miss Gladys was 30 years old.
  Fact 0.338: Cherry, Miss Gladys is from London, England, UK.
  Fact 0.339: Cherry, Miss Gladys boarded at Southampton.
  Fact 0.340: Cherry, Miss Gladys's destination is Vancouver, British Columbia, Canada.
  Fact 0.341: Cherry, Miss Gladys has 8 in the lifeboat or body column.
  Fact 0.342: Chevré, Mr. Paul Romaine Marie Léonce was 45 years old.
  Fact 0.343: Chevré, Mr. Paul Romaine Marie Léonce is from Paris, France.
  Fact 0.344: Chevré, Mr. Paul Romaine Marie Léonce boarded at Cherbourg.
  Fact 0.345: Chevré, Mr. Paul Romaine Marie Léonce's destination is Ottawa, Ontario, Canada.
  Fact 0.346: Chevré, Mr. Paul Romaine Marie Léonce has 7 in the lifeboat or body column.
  Fact 0.347: Chibnall, Mrs. Edith Martha Bowerman (née Barber) was 48 years old.
  Fact 0.348: Chibnall, Mrs. Edith Martha Bowerman (née Barber) is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.349: Chibnall, Mrs. Edith Martha Bowerman (née Barber) boarded at Southampton.
  Fact 0.350: Chibnall, Mrs. Edith Martha Bowerman (née Barber)'s destination is New York City.
  Fact 0.351: Chibnall, Mrs. Edith Martha Bowerman (née Barber) has 6 in the lifeboat or body column.
  Fact 0.352: Chisholm, Mr. Roderick Robert Crispin was 40 years old.
  Fact 0.353: Chisholm, Mr. Roderick Robert Crispin is from Belfast, Ireland, UK.
  Fact 0.354: Chisholm, Mr. Roderick Robert Crispin boarded at Belfast.
  Fact 0.355: Chisholm, Mr. Roderick Robert Crispin's destination is New York City.
  Fact 0.356: Clark, Mr. Walter Miller was 27 years old.
  Fact 0.357: Clark, Mr. Walter Miller is from Los Angeles, California, US.
  Fact 0.358: Clark, Mr. Walter Miller boarded at Cherbourg.
  Fact 0.359: Clark, Mr. Walter Miller's destination is Los Angeles, California, US.
  Fact 0.360: Clark, Mrs. Virginia Estelle (née McDowell) was 26 years old.
  Fact 0.361: Clark, Mrs. Virginia Estelle (née McDowell) is from Los Angeles, California, US.
  Fact 0.362: Clark, Mrs. Virginia Estelle (née McDowell) boarded at Cherbourg.
  Fact 0.363: Clark, Mrs. Virginia Estelle (née McDowell)'s destination is Los Angeles, California, US.
  Fact 0.364: Clifford, Mr. George Quincy was 40 years old.
  Fact 0.365: Clifford, Mr. George Quincy is from Stoughton, Massachusetts, US.
  Fact 0.366: Clifford, Mr. George Quincy boarded at Southampton.
  Fact 0.367: Clifford, Mr. George Quincy's destination is Stoughton, Massachusetts.
  Fact 0.368: Colley, Mr. Edward Pomeroy was 37 years old.
  Fact 0.369: Colley, Mr. Edward Pomeroy is from Dublin, Ireland, UK.
  Fact 0.370: Colley, Mr. Edward Pomeroy boarded at Southampton.
  Fact 0.371: Colley, Mr. Edward Pomeroy's destination is Vancouver, British Columbia, Canada.
  Fact 0.372: Compton, Mrs. Mary Eliza (née Ingersoll) was 64 years old.
  Fact 0.373: Compton, Mrs. Mary Eliza (née Ingersoll) is from Lakewood, New Jersey, US.
  Fact 0.374: Compton, Mrs. Mary Eliza (née Ingersoll) boarded at Cherbourg.
  Fact 0.375: Compton, Mrs. Mary Eliza (née Ingersoll)'s destination is Lakewood, New Jersey, US.
  Fact 0.376: Compton, Mrs. Mary Eliza (née Ingersoll) has 14 in the lifeboat or body column.
  Fact 0.377: Compton, Miss Sara Rebecca was 39 years old.
  Fact 0.378: Compton, Miss Sara Rebecca is from Lakewood, New Jersey, US.
  Fact 0.379: Compton, Miss Sara Rebecca boarded at Cherbourg.
  Fact 0.380: Compton, Miss Sara Rebecca's destination is Lakewood, New Jersey, US.
  Fact 0.381: Compton, Miss Sara Rebecca has 14 in the lifeboat or body column.
  Fact 0.382: Compton, Mr. Alexander Taylor Jr. was 37 years old.
  Fact 0.383: Compton, Mr. Alexander Taylor Jr. is from Lakewood, New Jersey, US.
  Fact 0.384: Compton, Mr. Alexander Taylor Jr. boarded at Cherbourg.
  Fact 0.385: Compton, Mr. Alexander Taylor Jr.'s destination is Lakewood, New Jersey, US.
  Fact 0.386: Cornell, Mrs. Malvina Helen (née Lamson) was 55 years old.
  Fact 0.387: Cornell, Mrs. Malvina Helen (née Lamson) is from New York City.
  Fact 0.388: Cornell, Mrs. Malvina Helen (née Lamson) boarded at Southampton.
  Fact 0.389: Cornell, Mrs. Malvina Helen (née Lamson)'s destination is New York City.
  Fact 0.390: Crafton, Mr. John Bertram was 59 years old.
  Fact 0.391: Crafton, Mr. John Bertram is from Roachdale, Indiana, US.
  Fact 0.392: Crafton, Mr. John Bertram boarded at Southampton.
  Fact 0.393: Crafton, Mr. John Bertram's destination is Roachdale, Indiana, US.
  Fact 0.394: Crosby, Captain Edward was 70 years old.
  Fact 0.395: Crosby, Captain Edward is from Milwaukee, Wisconsin, US.
  Fact 0.396: Crosby, Captain Edward boarded at Southampton.
  Fact 0.397: Crosby, Captain Edward's destination is Milwaukee, Wisconsin, US.
  Fact 0.398: Crosby, Captain Edward has 269 in the lifeboat or body column.
  Fact 0.399: Crosby, Mrs. Catherine Elizabeth (née Halstead) was 64 years old.
  Fact 0.400: Crosby, Mrs. Catherine Elizabeth (née Halstead) is from Milwaukee, Wisconsin, US.
  Fact 0.401: Crosby, Mrs. Catherine Elizabeth (née Halstead) boarded at Southampton.
  Fact 0.402: Crosby, Mrs. Catherine Elizabeth (née Halstead)'s destination is Milwaukee, Wisconsin, US.
  Fact 0.403: Crosby, Mrs. Catherine Elizabeth (née Halstead) has 7 in the lifeboat or body column.
  Fact 0.404: Crosby, Miss Harriette Rebecca was 39 years old.
  Fact 0.405: Crosby, Miss Harriette Rebecca is from Milwaukee, Wisconsin, US.
  Fact 0.406: Crosby, Miss Harriette Rebecca boarded at Southampton.
  Fact 0.407: Crosby, Miss Harriette Rebecca's destination is Milwaukee, Wisconsin, US.
  Fact 0.408: Crosby, Miss Harriette Rebecca has 7 in the lifeboat or body column.
  Fact 0.409: Cumings, Mr. John Bradley was 39 years old.
  Fact 0.410: Cumings, Mr. John Bradley is from New York City.
  Fact 0.411: Cumings, Mr. John Bradley boarded at Cherbourg.
  Fact 0.412: Cumings, Mr. John Bradley's destination is New York City.
  Fact 0.413: Cumings, Mrs. Florence Briggs (née Thayer) was 35 years old.
  Fact 0.414: Cumings, Mrs. Florence Briggs (née Thayer) is from New York City.
  Fact 0.415: Cumings, Mrs. Florence Briggs (née Thayer) boarded at Cherbourg.
  Fact 0.416: Cumings, Mrs. Florence Briggs (née Thayer)'s destination is New York, US.
  Fact 0.417: Daly, Mr. Peter Dennis was 51 years old.
  Fact 0.418: Daly, Mr. Peter Dennis is from Lima, Peru.
  Fact 0.419: Daly, Mr. Peter Dennis boarded at Southampton.
  Fact 0.420: Daly, Mr. Peter Dennis's destination is Lima, Peru.
  Fact 0.421: Daniel, Mr. Robert Williams was 27 years old.
  Fact 0.422: Daniel, Mr. Robert Williams is from Philadelphia, Pennsylvania, US.
  Fact 0.423: Daniel, Mr. Robert Williams boarded at Southampton.
  Fact 0.424: Daniel, Mr. Robert Williams's destination is Philadelphia, Pennsylvania, US.
  Fact 0.425: Davidson, Mr. Thornton was 31 years old.
  Fact 0.426: Davidson, Mr. Thornton is from Montreal, Quebec, Canada.
  Fact 0.427: Davidson, Mr. Thornton boarded at Southampton.
  Fact 0.428: Davidson, Mr. Thornton's destination is Montreal, Quebec, Canada.
  Fact 0.429: Davidson, Mrs. Orian (née Hays) was 27 years old.
  Fact 0.430: Davidson, Mrs. Orian (née Hays) is from Montreal, Quebec, Canada.
  Fact 0.431: Davidson, Mrs. Orian (née Hays) boarded at Southampton.
  Fact 0.432: Davidson, Mrs. Orian (née Hays)'s destination is Montreal, Quebec, Canada.
  Fact 0.433: Dick, Mr. Albert Adrian was 31 years old.
  Fact 0.434: Dick, Mr. Albert Adrian is from Calgary, Alberta, Canada.
  Fact 0.435: Dick, Mr. Albert Adrian boarded at Southampton.
  Fact 0.436: Dick, Mr. Albert Adrian's destination is Calgary, Alberta, Canada.
  Fact 0.437: Dick, Mrs. Vera (née Gillespie) was 17 years old.
  Fact 0.438: Dick, Mrs. Vera (née Gillespie) is from Calgary, Alberta, Canada.
  Fact 0.439: Dick, Mrs. Vera (née Gillespie) boarded at Southampton.
  Fact 0.440: Dick, Mrs. Vera (née Gillespie)'s destination is Calgary, Alberta, Canada.
  Fact 0.441: Dineston, Mrs. Nancy May Bowles was 28 years old.
  Fact 0.442: Dineston, Mrs. Nancy May Bowles is from London, England, UK.
  Fact 0.443: Dineston, Mrs. Nancy May Bowles boarded at Southampton.
  Fact 0.444: Dineston, Mrs. Nancy May Bowles's destination is New York City, USA.
  Fact 0.445: Dineston, George Samuel was 30 years old.
  Fact 0.446: Dineston, George Samuel is from London, England, UK.
  Fact 0.447: Dineston, George Samuel boarded at Southampton.
  Fact 0.448: Dineston, George Samuel's destination is New York City, USA.
  Fact 0.449: Dodge, Dr. Washington was 52 years old.
  Fact 0.450: Dodge, Dr. Washington is from San Francisco, California, US.
  Fact 0.451: Dodge, Dr. Washington boarded at Southampton.
  Fact 0.452: Dodge, Dr. Washington's destination is San Francisco, California, US.
  Fact 0.453: Dodge, Dr. Washington has 13 in the lifeboat or body column.
  Fact 0.454: Dodge, Mrs. Ruth (née Vidaver) was 34 years old.
  Fact 0.455: Dodge, Mrs. Ruth (née Vidaver) is from San Francisco, California, US.
  Fact 0.456: Dodge, Mrs. Ruth (née Vidaver) boarded at Southampton.
  Fact 0.457: Dodge, Mrs. Ruth (née Vidaver)'s destination is San Francisco, California, US.
  Fact 0.458: Dodge, Mrs. Ruth (née Vidaver) has 5 in the lifeboat or body column.
  Fact 0.459: Dodge, Master Washington, Jr. is from San Francisco, California, US.
  Fact 0.460: Dodge, Master Washington, Jr. boarded at Southampton.
  Fact 0.461: Dodge, Master Washington, Jr.'s destination is San Francisco, California, US.
  Fact 0.462: Dodge, Master Washington, Jr. has 5 in the lifeboat or body column.
  Fact 0.463: Douglas, Mr. Walter Donald was 50 years old.
  Fact 0.464: Douglas, Mr. Walter Donald is from Minneapolis, Minnesota, US.
  Fact 0.465: Douglas, Mr. Walter Donald boarded at Cherbourg.
  Fact 0.466: Douglas, Mr. Walter Donald's destination is Minneapolis, Minnesota, US.
  Fact 0.467: Douglas, Mr. Walter Donald has 62 in the lifeboat or body column.
  Fact 0.468: Douglas, Mrs. Mahala (née Dutton) was 48 years old.
  Fact 0.469: Douglas, Mrs. Mahala (née Dutton) is from Minneapolis, Minnesota, US.
  Fact 0.470: Douglas, Mrs. Mahala (née Dutton) boarded at Cherbourg.
  Fact 0.471: Douglas, Mrs. Mahala (née Dutton)'s destination is Minneapolis, Minnesota, US.
  Fact 0.472: Miss Berthe Leroy was 27 years old.
  Fact 0.473: Miss Berthe Leroy is listed as the maid.
  Fact 0.474: Miss Berthe Leroy is from Minneapolis, Minnesota, US.
  Fact 0.475: Miss Berthe Leroy boarded at Cherbourg.
  Fact 0.476: Miss Berthe Leroy's destination is Minneapolis, Minnesota, US.
  Fact 0.477: Douglas, Mrs. Mary Hélène (née Baxter) was 27 years old.
  Fact 0.478: Douglas, Mrs. Mary Hélène (née Baxter) is from Montreal, Quebec, Canada.
  Fact 0.479: Douglas, Mrs. Mary Hélène (née Baxter) boarded at Cherbourg.
  Fact 0.480: Douglas, Mrs. Mary Hélène (née Baxter)'s destination is Montreal, Quebec, Canada.
  Fact 0.481: Douglas, Mrs. Mary Hélène (née Baxter) has 6 in the lifeboat or body column.
  Fact 0.482: Duff Gordon, Sir Cosmo Edmund was 49 years old.
  Fact 0.483: Duff Gordon, Sir Cosmo Edmund is from London, England, UK.
  Fact 0.484: Duff Gordon, Sir Cosmo Edmund boarded at Cherbourg.
  Fact 0.485: Duff Gordon, Sir Cosmo Edmund's destination is New York City.
  Fact 0.486: Duff Gordon, Lucy Christiana, Lady (née Sutherland) was 48 years old.
  Fact 0.487: Duff Gordon, Lucy Christiana, Lady (née Sutherland) is from London, England, UK.
  Fact 0.488: Duff Gordon, Lucy Christiana, Lady (née Sutherland) boarded at Cherbourg.
  Fact 0.489: Duff Gordon, Lucy Christiana, Lady (née Sutherland)'s destination is New York City.
  Fact 0.490: Miss Laura Mabel Francatelli was 31 years old.
  Fact 0.491: Miss Laura Mabel Francatelli is listed as the secretary.
  Fact 0.492: Miss Laura Mabel Francatelli is from London, England, UK.
  Fact 0.493: Miss Laura Mabel Francatelli boarded at Cherbourg.
  Fact 0.494: Miss Laura Mabel Francatelli's destination is New York City.
  Fact 0.495: Dulles, Mr. William Crothers was 39 years old.
  Fact 0.496: Dulles, Mr. William Crothers is from Philadelphia, Pennsylvania, US.
  Fact 0.497: Dulles, Mr. William Crothers boarded at Cherbourg.
  Fact 0.498: Dulles, Mr. William Crothers's destination is Philadelphia, Pennsylvania, US.
  Fact 0.499: Dulles, Mr. William Crothers has 133 in the lifeboat or body column.
  Fact 0.500: Eustis, Miss Elizabeth Mussey was 54 years old.
  Fact 0.501: Eustis, Miss Elizabeth Mussey is from Brookline, Massachusetts, US.
  Fact 0.502: Eustis, Miss Elizabeth Mussey boarded at Cherbourg.
  Fact 0.503: Eustis, Miss Elizabeth Mussey's destination is Brookline, Massachusetts, US.
  Fact 0.504: Evans, Miss Edith Corse was 36 years old.
  Fact 0.505: Evans, Miss Edith Corse is from New York City.
  Fact 0.506: Evans, Miss Edith Corse boarded at Cherbourg.
  Fact 0.507: Evans, Miss Edith Corse's destination is New York City.
  Fact 0.508: Flegenheim, Mrs. Antoinette (née Wendt) was 48 years old.
  Fact 0.509: Flegenheim, Mrs. Antoinette (née Wendt) is from New York City.
  Fact 0.510: Flegenheim, Mrs. Antoinette (née Wendt) boarded at Cherbourg.
  Fact 0.511: Flegenheim, Mrs. Antoinette (née Wendt)'s destination is New York City.
  Fact 0.512: Flegenheim, Mrs. Antoinette (née Wendt) has 7 in the lifeboat or body column.
  Fact 0.513: Flynn, Mr. John Irwin was 36 years old.
  Fact 0.514: Flynn, Mr. John Irwin is from Brooklyn, New York, US.
  Fact 0.515: Flynn, Mr. John Irwin boarded at Southampton.
  Fact 0.516: Flynn, Mr. John Irwin's destination is Brooklyn, New York, US.
  Fact 0.517: Flynn, Mr. John Irwin has 5 in the lifeboat or body column.
  Fact 0.518: Foreman, Mr. Benjamin Laventall was 30 years old.
  Fact 0.519: Foreman, Mr. Benjamin Laventall is from New York City.
  Fact 0.520: Foreman, Mr. Benjamin Laventall boarded at Southampton.
  Fact 0.521: Foreman, Mr. Benjamin Laventall's destination is New York City.
  Fact 0.522: Fortune, Mr. Mark was 64 years old.
  Fact 0.523: Fortune, Mr. Mark is from Winnipeg, Manitoba, Canada.
  Fact 0.524: Fortune, Mr. Mark boarded at Southampton.
  Fact 0.525: Fortune, Mr. Mark's destination is Winnipeg, Manitoba, Canada.
  Fact 0.526: Fortune, Mrs. Mary (née McDougald) was 60 years old.
  Fact 0.527: Fortune, Mrs. Mary (née McDougald) is from Winnipeg, Manitoba, Canada.
  Fact 0.528: Fortune, Mrs. Mary (née McDougald) boarded at Southampton.
  Fact 0.529: Fortune, Mrs. Mary (née McDougald)'s destination is Winnipeg, Manitoba, Canada.
  Fact 0.530: Fortune, Mrs. Mary (née McDougald) has 10 in the lifeboat or body column.
  Fact 0.531: Fortune, Miss Ethel Flora was 28 years old.
  Fact 0.532: Fortune, Miss Ethel Flora is from Winnipeg, Manitoba, Canada.
  Fact 0.533: Fortune, Miss Ethel Flora boarded at Southampton.
  Fact 0.534: Fortune, Miss Ethel Flora's destination is Winnipeg, Manitoba, Canada.
  Fact 0.535: Fortune, Miss Ethel Flora has 10 in the lifeboat or body column.
  Fact 0.536: Fortune, Miss Alice Elizabeth was 24 years old.
  Fact 0.537: Fortune, Miss Alice Elizabeth is from Winnipeg, Manitoba, Canada.
  Fact 0.538: Fortune, Miss Alice Elizabeth boarded at Southampton.
  Fact 0.539: Fortune, Miss Alice Elizabeth's destination is Winnipeg, Manitoba, Canada.
  Fact 0.540: Fortune, Miss Alice Elizabeth has 10 in the lifeboat or body column.
  Fact 0.541: Fortune, Miss Mabel Helen was 23 years old.
  Fact 0.542: Fortune, Miss Mabel Helen is from Winnipeg, Manitoba, Canada.
  Fact 0.543: Fortune, Miss Mabel Helen boarded at Southampton.
  Fact 0.544: Fortune, Miss Mabel Helen's destination is Winnipeg, Manitoba, Canada.
  Fact 0.545: Fortune, Miss Mabel Helen has 10 in the lifeboat or body column.
  Fact 0.546: Fortune, Mr. Charles Alexander was 19 years old.
  Fact 0.547: Fortune, Mr. Charles Alexander is from Winnipeg, Manitoba, Canada.
  Fact 0.548: Fortune, Mr. Charles Alexander boarded at Southampton.
  Fact 0.549: Fortune, Mr. Charles Alexander's destination is Winnipeg, Manitoba, Canada.
  Fact 0.550: Franklin, Mr. Thomas Parnham was 37 years old.
  Fact 0.551: Franklin, Mr. Thomas Parnham is from London, England, UK.
  Fact 0.552: Franklin, Mr. Thomas Parnham boarded at Southampton.
  Fact 0.553: Franklin, Mr. Thomas Parnham's destination is New York City.
  Fact 0.554: Frauenthal, Dr. Henry William was 49 years old.
  Fact 0.555: Frauenthal, Dr. Henry William is from London, England, UK.
  Fact 0.556: Frauenthal, Dr. Henry William boarded at Southampton.
  Fact 0.557: Frauenthal, Dr. Henry William's destination is New York City.
  Fact 0.558: Frauenthal, Dr. Henry William has 5 in the lifeboat or body column.
  Fact 0.559: Frauenthal, Mrs. Clara (née Heinsheimer) was 42 years old.
  Fact 0.560: Frauenthal, Mrs. Clara (née Heinsheimer) is from London, England, UK.
  Fact 0.561: Frauenthal, Mrs. Clara (née Heinsheimer) boarded at Southampton.
  Fact 0.562: Frauenthal, Mrs. Clara (née Heinsheimer)'s destination is New York City.
  Fact 0.563: Frauenthal, Mrs. Clara (née Heinsheimer) has 5 in the lifeboat or body column.
  Fact 0.564: Frauenthal, Mr. Isaac Gerald was 43 years old.
  Fact 0.565: Frauenthal, Mr. Isaac Gerald is from London, England, UK.
  Fact 0.566: Frauenthal, Mr. Isaac Gerald boarded at Southampton.
  Fact 0.567: Frauenthal, Mr. Isaac Gerald's destination is New York City.
  Fact 0.568: Frauenthal, Mr. Isaac Gerald has 5 in the lifeboat or body column.
  Fact 0.569: Frölicher, Mr. Maximilian Josef was 60 years old.
  Fact 0.570: Frölicher, Mr. Maximilian Josef is from Zürich, Switzerland.
  Fact 0.571: Frölicher, Mr. Maximilian Josef boarded at Cherbourg.
  Fact 0.572: Frölicher, Mr. Maximilian Josef's destination is New York City.
  Fact 0.573: Frölicher, Mr. Maximilian Josef has 5 in the lifeboat or body column.
  Fact 0.574: Frölicher, Mrs. Margaretha Emerentia (née Stehli) was 48 years old.
  Fact 0.575: Frölicher, Mrs. Margaretha Emerentia (née Stehli) is from Zürich, Switzerland.
  Fact 0.576: Frölicher, Mrs. Margaretha Emerentia (née Stehli) boarded at Cherbourg.
  Fact 0.577: Frölicher, Mrs. Margaretha Emerentia (née Stehli)'s destination is New York City.
  Fact 0.578: Frölicher, Mrs. Margaretha Emerentia (née Stehli) has 5 in the lifeboat or body column.
  Fact 0.579: Frölicher-Stehli, Miss Hedwig Margaritha was 22 years old.
  Fact 0.580: Frölicher-Stehli, Miss Hedwig Margaritha is from Zürich, Switzerland.
  Fact 0.581: Frölicher-Stehli, Miss Hedwig Margaritha boarded at Cherbourg.
  Fact 0.582: Frölicher-Stehli, Miss Hedwig Margaritha's destination is New York City.
  Fact 0.583: Frölicher-Stehli, Miss Hedwig Margaritha has 5 in the lifeboat or body column.
  Fact 0.584: Futrelle, Mr. Jacques Heath was 37 years old.
  Fact 0.585: Futrelle, Mr. Jacques Heath is from Scituate, Massachusetts, US.
  Fact 0.586: Futrelle, Mr. Jacques Heath boarded at Southampton.
  Fact 0.587: Futrelle, Mr. Jacques Heath's destination is Scituate, Massachusetts, US.
  Fact 0.588: Futrelle, Mrs. Lily May (née Peel) was 35 years old.
  Fact 0.589: Futrelle, Mrs. Lily May (née Peel) is from Scituate, Massachusetts, US.
  Fact 0.590: Futrelle, Mrs. Lily May (née Peel) boarded at Southampton.
  Fact 0.591: Futrelle, Mrs. Lily May (née Peel)'s destination is Scituate, Massachusetts, US.
  Fact 0.592: Gee, Mr. Arthur H. was 47 years old.
  Fact 0.593: Gee, Mr. Arthur H. is from St. Annes-on-Sea, Lancashire, England, UK.
  Fact 0.594: Gee, Mr. Arthur H. boarded at Southampton.
  Fact 0.595: Gee, Mr. Arthur H.'s destination is Mexico City, Mexico.
  Fact 0.596: Gee, Mr. Arthur H. has 275 in the lifeboat or body column.
  Fact 0.597: Gibson, Mrs. Pauline Caroline (née Boeson) was 44 years old.
  Fact 0.598: Gibson, Mrs. Pauline Caroline (née Boeson) is from New York City.
  Fact 0.599: Gibson, Mrs. Pauline Caroline (née Boeson) boarded at Cherbourg.
  Fact 0.600: Gibson, Mrs. Pauline Caroline (née Boeson)'s destination is New York City.
  Fact 0.601: Gibson, Mrs. Pauline Caroline (née Boeson) has 7 in the lifeboat or body column.
  Fact 0.602: Gibson, Miss Dorothy Winifred was 22 years old.
  Fact 0.603: Gibson, Miss Dorothy Winifred is from New York City.
  Fact 0.604: Gibson, Miss Dorothy Winifred boarded at Cherbourg.
  Fact 0.605: Gibson, Miss Dorothy Winifred's destination is New York City.
  Fact 0.606: Gibson, Miss Dorothy Winifred has 7 in the lifeboat or body column.
  Fact 0.607: Goldenberg, Mr. Samuel L. was 47 years old.
  Fact 0.608: Goldenberg, Mr. Samuel L. is from Paris, France.
  Fact 0.609: Goldenberg, Mr. Samuel L. boarded at Cherbourg.
  Fact 0.610: Goldenberg, Mr. Samuel L.'s destination is New York City.
  Fact 0.611: Goldenberg, Mr. Samuel L. has 5 in the lifeboat or body column.
  Fact 0.612: Goldenberg, Mrs. Nella (née Wiggins) was 40 years old.
  Fact 0.613: Goldenberg, Mrs. Nella (née Wiggins) is from Paris, France.
  Fact 0.614: Goldenberg, Mrs. Nella (née Wiggins) boarded at Cherbourg.
  Fact 0.615: Goldenberg, Mrs. Nella (née Wiggins)'s destination is New York City.
  Fact 0.616: Goldenberg, Mrs. Nella (née Wiggins) has 5 in the lifeboat or body column.
  Fact 0.617: Goldschmidt, Mr. George B. was 71 years old.
  Fact 0.618: Goldschmidt, Mr. George B. is from New York City.
  Fact 0.619: Goldschmidt, Mr. George B. boarded at Cherbourg.
  Fact 0.620: Goldschmidt, Mr. George B.'s destination is New York City.
  Fact 0.621: Gracie IV, Colonel Archibald was 53 years old.
  Fact 0.622: Gracie IV, Colonel Archibald is from Washington, D.C., US.
  Fact 0.623: Gracie IV, Colonel Archibald boarded at Southampton.
  Fact 0.624: Gracie IV, Colonel Archibald's destination is Washington, D.C., US.
  Fact 0.625: Graham, Mr. George Edward was 38 years old.
  Fact 0.626: Graham, Mr. George Edward is from Winnipeg, Manitoba, Canada.
  Fact 0.627: Graham, Mr. George Edward boarded at Southampton.
  Fact 0.628: Graham, Mr. George Edward's destination is Winnipeg, Manitoba, Canada.
  Fact 0.629: Graham, Mr. George Edward has 147 in the lifeboat or body column.
  Fact 0.630: Graham, Mrs. Edith Ware (née Junkins) was 59 years old.
  Fact 0.631: Graham, Mrs. Edith Ware (née Junkins) is from Greenwich, Connecticut, US.
  Fact 0.632: Graham, Mrs. Edith Ware (née Junkins) boarded at Southampton.
  Fact 0.633: Graham, Mrs. Edith Ware (née Junkins)'s destination is Greenwich, Connecticut, US.
  Fact 0.634: Graham, Miss Margaret Edith was 19 years old.
  Fact 0.635: Graham, Miss Margaret Edith is from Greenwich, Connecticut, US.
  Fact 0.636: Graham, Miss Margaret Edith boarded at Southampton.
  Fact 0.637: Graham, Miss Margaret Edith's destination is Greenwich, Connecticut, US.
  Fact 0.638: Miss Elizabeth Weed Shutes was 40 years old.
  Fact 0.639: Miss Elizabeth Weed Shutes is listed as the governess.
  Fact 0.640: Miss Elizabeth Weed Shutes is from Greenwich, Connecticut, US.
  Fact 0.641: Miss Elizabeth Weed Shutes boarded at Southampton.
  Fact 0.642: Miss Elizabeth Weed Shutes's destination is Greenwich, Connecticut, US.
  Fact 0.643: Greenfield, Mrs. Blanche (née Strouse) was 45 years old.
  Fact 0.644: Greenfield, Mrs. Blanche (née Strouse) is from New York City.
  Fact 0.645: Greenfield, Mrs. Blanche (née Strouse) boarded at Cherbourg.
  Fact 0.646: Greenfield, Mrs. Blanche (née Strouse)'s destination is New York City.
  Fact 0.647: Greenfield, Mrs. Blanche (née Strouse) has 7 in the lifeboat or body column.
  Fact 0.648: Greenfield, Mr. William Bertram was 23 years old.
  Fact 0.649: Greenfield, Mr. William Bertram is from New York City.
  Fact 0.650: Greenfield, Mr. William Bertram boarded at Cherbourg.
  Fact 0.651: Greenfield, Mr. William Bertram's destination is New York City.
  Fact 0.652: Greenfield, Mr. William Bertram has 7 in the lifeboat or body column.
  Fact 0.653: Guggenheim, Mr. Benjamin was 46 years old.
  Fact 0.654: Guggenheim, Mr. Benjamin is from Paris, France.
  Fact 0.655: Guggenheim, Mr. Benjamin boarded at Cherbourg.
  Fact 0.656: Guggenheim, Mr. Benjamin's destination is New York City.
  Fact 0.657: Mr. Victor Giglio was 24 years old.
  Fact 0.658: Mr. Victor Giglio is listed as the valet.
  Fact 0.659: Mr. Victor Giglio is from Paris, France.
  Fact 0.660: Mr. Victor Giglio boarded at Cherbourg.
  Fact 0.661: Mr. Victor Giglio's destination is New York City.
  Fact 0.662: Mr. René Pernot was 39 years old.
  Fact 0.663: Mr. René Pernot is listed as the chauffeur.
  Fact 0.664: Mr. René Pernot is from Paris, France.
  Fact 0.665: Mr. René Pernot boarded at Cherbourg.
  Fact 0.666: Mr. René Pernot's destination is New York City.
  Fact 0.667: Harder, Mr. George Achilles was 25 years old.
  Fact 0.668: Harder, Mr. George Achilles is from New York City.
  Fact 0.669: Harder, Mr. George Achilles boarded at Cherbourg.
  Fact 0.670: Harder, Mr. George Achilles's destination is New York City.
  Fact 0.671: Harder, Mr. George Achilles has 5 in the lifeboat or body column.
  Fact 0.672: Harder, Mrs. Dorothy (née Annan) was 21 years old.
  Fact 0.673: Harder, Mrs. Dorothy (née Annan) is from New York City.
  Fact 0.674: Harder, Mrs. Dorothy (née Annan) boarded at Cherbourg.
  Fact 0.675: Harder, Mrs. Dorothy (née Annan)'s destination is New York City.
  Fact 0.676: Harder, Mrs. Dorothy (née Annan) has 5 in the lifeboat or body column.
  Fact 0.677: Harper, Mr. Henry Sleeper was 48 years old.
  Fact 0.678: Harper, Mr. Henry Sleeper is from New York City.
  Fact 0.679: Harper, Mr. Henry Sleeper boarded at Cherbourg.
  Fact 0.680: Harper, Mr. Henry Sleeper's destination is New York City.
  Fact 0.681: Mr. Hammad Hassab was 27 years old.
  Fact 0.682: Mr. Hammad Hassab is listed as the dragoman.
  Fact 0.683: Mr. Hammad Hassab is from Cairo, Egypt.
  Fact 0.684: Mr. Hammad Hassab boarded at Cherbourg.
  Fact 0.685: Mr. Hammad Hassab's destination is New York City.
  Fact 0.686: Harper, Mrs. Myna (née Haxtun) was 49 years old.
  Fact 0.687: Harper, Mrs. Myna (née Haxtun) is from New York City.
  Fact 0.688: Harper, Mrs. Myna (née Haxtun) boarded at Cherbourg.
  Fact 0.689: Harper, Mrs. Myna (née Haxtun)'s destination is New York City.
  Fact 0.690: Harris, Mr. Henry Birkhardt was 45 years old.
  Fact 0.691: Harris, Mr. Henry Birkhardt is from New York City.
  Fact 0.692: Harris, Mr. Henry Birkhardt boarded at Southampton.
  Fact 0.693: Harris, Mr. Henry Birkhardt's destination is New York City.
  Fact 0.694: Harris, Mrs. Irene (née Wallach) was 35 years old.
  Fact 0.695: Harris, Mrs. Irene (née Wallach) is from New York City.
  Fact 0.696: Harris, Mrs. Irene (née Wallach) boarded at Southampton.
  Fact 0.697: Harris, Mrs. Irene (née Wallach)'s destination is New York City.
  Fact 0.698: Hawksford, Mr. Walter James was 45 years old.
  Fact 0.699: Hawksford, Mr. Walter James is from Kingston, Surrey, England, UK.
  Fact 0.700: Hawksford, Mr. Walter James boarded at Southampton.
  Fact 0.701: Hawksford, Mr. Walter James's destination is New York City.
  Fact 0.702: Hays, Mr. Charles Melville was 55 years old.
  Fact 0.703: Hays, Mr. Charles Melville is from Montreal, Quebec, Canada.
  Fact 0.704: Hays, Mr. Charles Melville boarded at Southampton.
  Fact 0.705: Hays, Mr. Charles Melville's destination is Montreal, Quebec, Canada.
  Fact 0.706: Hays, Mr. Charles Melville has 307 in the lifeboat or body column.
  Fact 0.707: Mr. Vivian Ponsonby Payne was 22 years old.
  Fact 0.708: Mr. Vivian Ponsonby Payne is listed as the clerk.
  Fact 0.709: Mr. Vivian Ponsonby Payne is from Montreal, Quebec, Canada.
  Fact 0.710: Mr. Vivian Ponsonby Payne boarded at Southampton.
  Fact 0.711: Mr. Vivian Ponsonby Payne's destination is Montreal, Quebec, Canada.
  Fact 0.712: Hays, Mrs. Clara Jennings (née Grigg) was 52 years old.
  Fact 0.713: Hays, Mrs. Clara Jennings (née Grigg) is from Montreal, Quebec, Canada.
  Fact 0.714: Hays, Mrs. Clara Jennings (née Grigg) boarded at Southampton.
  Fact 0.715: Hays, Mrs. Clara Jennings (née Grigg)'s destination is Montreal, Quebec, Canada.
  Fact 0.716: Miss Mary Anne Perreault was 33 years old.
  Fact 0.717: Miss Mary Anne Perreault is listed as the maid.
  Fact 0.718: Miss Mary Anne Perreault is from Montreal, Quebec, Canada.
  Fact 0.719: Miss Mary Anne Perreault boarded at Southampton.
  Fact 0.720: Miss Mary Anne Perreault's destination is Montreal, Quebec, Canada.
  Fact 0.721: Hays, Miss Margaret Bechstein was 24 years old.
  Fact 0.722: Hays, Miss Margaret Bechstein is from New York City.
  Fact 0.723: Hays, Miss Margaret Bechstein boarded at Cherbourg.
  Fact 0.724: Hays, Miss Margaret Bechstein's destination is New York City.
  Fact 0.725: Hays, Miss Margaret Bechstein has 7 in the lifeboat or body column.
  Fact 0.726: Head, Mr. Christopher was 42 years old.
  Fact 0.727: Head, Mr. Christopher is from Chelsea, London, England, UK.
  Fact 0.728: Head, Mr. Christopher boarded at Southampton.
  Fact 0.729: Head, Mr. Christopher's destination is New York City.
  Fact 0.730: Hilliard, Mr. Herbert Henry was 44 years old.
  Fact 0.731: Hilliard, Mr. Herbert Henry is from Brighton, Massachusetts, US.
  Fact 0.732: Hilliard, Mr. Herbert Henry boarded at Southampton.
  Fact 0.733: Hilliard, Mr. Herbert Henry's destination is Brighton, Massachusetts, US.
  Fact 0.734: Hipkins, Mr. William Edward was 55 years old.
  Fact 0.735: Hipkins, Mr. William Edward is from Birmingham, West Midlands, England, UK.
  Fact 0.736: Hipkins, Mr. William Edward boarded at Southampton.
  Fact 0.737: Hipkins, Mr. William Edward's destination is New York City.
  Fact 0.738: Hippach, Mrs. Ida Sophia (née Fischer) was 44 years old.
  Fact 0.739: Hippach, Mrs. Ida Sophia (née Fischer) is from Chicago, Illinois, US.
  Fact 0.740: Hippach, Mrs. Ida Sophia (née Fischer) boarded at Cherbourg.
  Fact 0.741: Hippach, Mrs. Ida Sophia (née Fischer)'s destination is Chicago, Illinois, US.
  Fact 0.742: Hippach, Miss Jean Gertrude was 17 years old.
  Fact 0.743: Hippach, Miss Jean Gertrude is from Chicago, Illinois, US.
  Fact 0.744: Hippach, Miss Jean Gertrude boarded at Cherbourg.
  Fact 0.745: Hippach, Miss Jean Gertrude's destination is Chicago, Illinois, US.
  Fact 0.746: Hogeboom, Mrs. Anna Louisa (née Andrews) was 51 years old.
  Fact 0.747: Hogeboom, Mrs. Anna Louisa (née Andrews) is from Hudson, New York, US.
  Fact 0.748: Hogeboom, Mrs. Anna Louisa (née Andrews) boarded at Cherbourg.
  Fact 0.749: Hogeboom, Mrs. Anna Louisa (née Andrews)'s destination is Hudson, New York, US.
  Fact 0.750: Hogeboom, Mrs. Anna Louisa (née Andrews) has 10 in the lifeboat or body column.
  Fact 0.751: Holverson, Mr. Alexander Oskar was 42 years old.
  Fact 0.752: Holverson, Mr. Alexander Oskar is from New York City.
  Fact 0.753: Holverson, Mr. Alexander Oskar boarded at Southampton.
  Fact 0.754: Holverson, Mr. Alexander Oskar's destination is New York City.
  Fact 0.755: Holverson, Mr. Alexander Oskar has 38 in the lifeboat or body column.
  Fact 0.756: Holverson, Mrs. Mary Aline (née Towner) was 35 years old.
  Fact 0.757: Holverson, Mrs. Mary Aline (née Towner) is from New York City.
  Fact 0.758: Holverson, Mrs. Mary Aline (née Towner) boarded at Southampton.
  Fact 0.759: Holverson, Mrs. Mary Aline (née Towner)'s destination is New York City.
  Fact 0.760: Holverson, Mrs. Mary Aline (née Towner) has 8 in the lifeboat or body column.
  Fact 0.761: Homer, Mr. Harry (alias E. Haven) was 40 years old.
  Fact 0.762: Homer, Mr. Harry (alias E. Haven) is from Indianapolis, Indiana, US.
  Fact 0.763: Homer, Mr. Harry (alias E. Haven) boarded at Southampton.
  Fact 0.764: Homer, Mr. Harry (alias E. Haven)'s destination is Indianapolis, Indiana, US.
  Fact 0.765: Homer, Mr. Harry (alias E. Haven) has 15 in the lifeboat or body column.
  Fact 0.766: Hoyt, Mr. Frederick Maxfield was 35 years old.
  Fact 0.767: Hoyt, Mr. Frederick Maxfield is from New York City.
  Fact 0.768: Hoyt, Mr. Frederick Maxfield boarded at Southampton.
  Fact 0.769: Hoyt, Mr. Frederick Maxfield's destination is Stamford, Connecticut, US.
  Fact 0.770: Hoyt, Mrs. Jane Anne (née Forby) was 31 years old.
  Fact 0.771: Hoyt, Mrs. Jane Anne (née Forby) is from New York City.
  Fact 0.772: Hoyt, Mrs. Jane Anne (née Forby) boarded at Southampton.
  Fact 0.773: Hoyt, Mrs. Jane Anne (née Forby)'s destination is Stamford, Connecticut, US.
  Fact 0.774: Hoyt, Mr. William Fisher was 42 years old.
  Fact 0.775: Hoyt, Mr. William Fisher is from New York City.
  Fact 0.776: Hoyt, Mr. William Fisher boarded at Cherbourg.
  Fact 0.777: Hoyt, Mr. William Fisher's destination is New York City.
  Fact 0.778: Hoyt, Mr. William Fisher has 14 in the lifeboat or body column.
  Fact 0.779: Isham, Miss Ann Elizabeth was 50 years old.
  Fact 0.780: Isham, Miss Ann Elizabeth is from Chicago, Illinois, US.
  Fact 0.781: Isham, Miss Ann Elizabeth boarded at Cherbourg.
  Fact 0.782: Isham, Miss Ann Elizabeth's destination is Chicago, Illinois, US.
  Fact 0.783: Ismay, Mr. Joseph Bruce was 49 years old.
  Fact 0.784: Ismay, Mr. Joseph Bruce is from Liverpool, Merseyside, England, UK.
  Fact 0.785: Ismay, Mr. Joseph Bruce boarded at Southampton.
  Fact 0.786: Ismay, Mr. Joseph Bruce's destination is New York City.
  Fact 0.787: Mr. John Richard Fry was 39 years old.
  Fact 0.788: Mr. John Richard Fry is listed as the valet.
  Fact 0.789: Mr. John Richard Fry is from Liverpool, Merseyside, England, UK.
  Fact 0.790: Mr. John Richard Fry boarded at Southampton.
  Fact 0.791: Mr. John Richard Fry's destination is New York City.
  Fact 0.792: Mr. William Henry Harrison was 45 years old.
  Fact 0.793: Mr. William Henry Harrison is listed as the secretary.
  Fact 0.794: Mr. William Henry Harrison is from Wallasey, Merseyside, England, UK.
  Fact 0.795: Mr. William Henry Harrison boarded at Southampton.
  Fact 0.796: Mr. William Henry Harrison's destination is New York City.
  Fact 0.797: Mr. William Henry Harrison has 110 in the lifeboat or body column.
  Fact 0.798: Johnson, Mrs. Madeline Eliza (née Austin) was 40 years old.
  Fact 0.799: Johnson, Mrs. Madeline Eliza (née Austin) is from London, England, UK.
  Fact 0.800: Johnson, Mrs. Madeline Eliza (née Austin) boarded at Southampton.
  Fact 0.801: Johnson, Mrs. Madeline Eliza (née Austin)'s destination is New York, New York, US.
  Fact 0.802: Johnson, Mrs. Madeline Eliza (née Austin) has 6 in the lifeboat or body column.
  Fact 0.803: Johnson (later changed to Langston), Mr. Samuel James was 10 years old.
  Fact 0.804: Johnson (later changed to Langston), Mr. Samuel James is from London, England, UK.
  Fact 0.805: Johnson (later changed to Langston), Mr. Samuel James boarded at Southampton.
  Fact 0.806: Johnson (later changed to Langston), Mr. Samuel James's destination is New York, New York, US.
  Fact 0.807: Jones, Mr. Charles Cresson was 46 years old.
  Fact 0.808: Jones, Mr. Charles Cresson is from Bennington, Vermont, US.
  Fact 0.809: Jones, Mr. Charles Cresson boarded at Southampton.
  Fact 0.810: Jones, Mr. Charles Cresson's destination is Bennington, Vermont, US.
  Fact 0.811: Jones, Mr. Charles Cresson has 80 in the lifeboat or body column.
  Fact 0.812: Julian, Mr. Henry Forbes was 50 years old.
  Fact 0.813: Julian, Mr. Henry Forbes is from Torquay, Devon, England, UK.
  Fact 0.814: Julian, Mr. Henry Forbes boarded at Southampton.
  Fact 0.815: Julian, Mr. Henry Forbes's destination is San Francisco, US.
  Fact 0.816: Kent, Mr. Edward Austin was 58 years old.
  Fact 0.817: Kent, Mr. Edward Austin is from Buffalo, New York, US.
  Fact 0.818: Kent, Mr. Edward Austin boarded at Cherbourg.
  Fact 0.819: Kent, Mr. Edward Austin's destination is Buffalo, New York, US.
  Fact 0.820: Kent, Mr. Edward Austin has 258 in the lifeboat or body column.
  Fact 0.821: Kenyon, Mr. Frederick R. was 41 years old.
  Fact 0.822: Kenyon, Mr. Frederick R. is from Pittsburgh, Pennsylvania, US.
  Fact 0.823: Kenyon, Mr. Frederick R. boarded at Southampton.
  Fact 0.824: Kenyon, Mr. Frederick R.'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.825: Kenyon, Mrs. Marion (née Stauffer) was 31 years old.
  Fact 0.826: Kenyon, Mrs. Marion (née Stauffer) is from Pittsburgh, Pennsylvania, US.
  Fact 0.827: Kenyon, Mrs. Marion (née Stauffer) boarded at Southampton.
  Fact 0.828: Kenyon, Mrs. Marion (née Stauffer)'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.829: Kenyon, Mrs. Marion (née Stauffer) has 8 in the lifeboat or body column.
  Fact 0.830: Kimball, Mr. Edwin Nelson Jr. was 42 years old.
  Fact 0.831: Kimball, Mr. Edwin Nelson Jr. is from Boston, Massachusetts, US.
  Fact 0.832: Kimball, Mr. Edwin Nelson Jr. boarded at Southampton.
  Fact 0.833: Kimball, Mr. Edwin Nelson Jr.'s destination is Boston, Massachusetts, US.
  Fact 0.834: Kimball, Mr. Edwin Nelson Jr. has 5 in the lifeboat or body column.
  Fact 0.835: Kimball, Mrs. Gertrude (née Parsons) was 45 years old.
  Fact 0.836: Kimball, Mrs. Gertrude (née Parsons) is from Boston, Massachusetts, US.
  Fact 0.837: Kimball, Mrs. Gertrude (née Parsons) boarded at Southampton.
  Fact 0.838: Kimball, Mrs. Gertrude (née Parsons)'s destination is Boston, Massachusetts, US.
  Fact 0.839: Kimball, Mrs. Gertrude (née Parsons) has 5 in the lifeboat or body column.
  Fact 0.840: Klaber, Mr. Herman was 45 years old.
  Fact 0.841: Klaber, Mr. Herman is from Portland, Oregon, US.
  Fact 0.842: Klaber, Mr. Herman boarded at Southampton.
  Fact 0.843: Klaber, Mr. Herman's destination is Portland, Oregon, US.
  Fact 0.844: Langston, Ms. Maxine Jane was 25 years old.
  Fact 0.845: Langston, Ms. Maxine Jane is from Duncansville, Pennsylvania, US.
  Fact 0.846: Langston, Ms. Maxine Jane boarded at Southampton.
  Fact 0.847: Langston, Ms. Maxine Jane's destination is Duncansivlle, Pennsylvania, US.
  Fact 0.848: Lambert-Williams, Mr. Fletcher Fellows was 43 years old.
  Fact 0.849: Lambert-Williams, Mr. Fletcher Fellows is from London, England, UK.
  Fact 0.850: Lambert-Williams, Mr. Fletcher Fellows boarded at Southampton.
  Fact 0.851: Lambert-Williams, Mr. Fletcher Fellows's destination is Newark, New Jersey, US.
  Fact 0.852: Leader, Dr. Alice (née Farnham) was 49 years old.
  Fact 0.853: Leader, Dr. Alice (née Farnham) is from New York City.
  Fact 0.854: Leader, Dr. Alice (née Farnham) boarded at Southampton.
  Fact 0.855: Leader, Dr. Alice (née Farnham)'s destination is New York City.
  Fact 0.856: Leader, Dr. Alice (née Farnham) has 8 in the lifeboat or body column.
  Fact 0.857: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) was 27 years old.
  Fact 0.858: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) is from Vancouver, British Columbia, Canada.
  Fact 0.859: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) boarded at Southampton.
  Fact 0.860: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes)'s destination is Vancouver, British Columbia, Canada.
  Fact 0.861: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) has 8 in the lifeboat or body column.
  Fact 0.862: Miss Roberta Elizabeth Mary "Cissy" Maioni was 20 years old.
  Fact 0.863: Miss Roberta Elizabeth Mary "Cissy" Maioni is listed as the maid.
  Fact 0.864: Miss Roberta Elizabeth Mary "Cissy" Maioni is from Vancouver, British Columbia, Canada.
  Fact 0.865: Miss Roberta Elizabeth Mary "Cissy" Maioni boarded at Southampton.
  Fact 0.866: Miss Roberta Elizabeth Mary "Cissy" Maioni's destination is Vancouver, British Columbia, Canada.
  Fact 0.867: Miss Roberta Elizabeth Mary "Cissy" Maioni has 8 in the lifeboat or body column.
  Fact 0.868: Lewy, Mr. Ervin G. was 31 years old.
  Fact 0.869: Lewy, Mr. Ervin G. is from Chicago, Illinois, US.
  Fact 0.870: Lewy, Mr. Ervin G. boarded at Cherbourg.
  Fact 0.871: Lewy, Mr. Ervin G.'s destination is Chicago, Illinois, US.
  Fact 0.872: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) was 42 years old.
  Fact 0.873: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) is from Jordanstorp, Södermanland, Sweden.
  Fact 0.874: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) boarded at Southampton.
  Fact 0.875: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey)'s destination is New York City.
  Fact 0.876: Lindström, Mrs. Sigrid (née Posse) was 55 years old.
  Fact 0.877: Lindström, Mrs. Sigrid (née Posse) is from Stockholm, Sweden.
  Fact 0.878: Lindström, Mrs. Sigrid (née Posse) boarded at Cherbourg.
  Fact 0.879: Lindström, Mrs. Sigrid (née Posse)'s destination is New York City.
  Fact 0.880: Lindström, Mrs. Sigrid (née Posse) has 6 in the lifeboat or body column.
  Fact 0.881: Lines, Mrs. Elizabeth Lindsey (née James) was 50 years old.
  Fact 0.882: Lines, Mrs. Elizabeth Lindsey (née James) is from Paris, France.
  Fact 0.883: Lines, Mrs. Elizabeth Lindsey (née James) boarded at Cherbourg.
  Fact 0.884: Lines, Mrs. Elizabeth Lindsey (née James)'s destination is Hanover, New Hampshire, US.
  Fact 0.885: Lines, Mrs. Elizabeth Lindsey (née James) has 9 in the lifeboat or body column.
  Fact 0.886: Lines, Miss Mary Conover was 16 years old.
  Fact 0.887: Lines, Miss Mary Conover is from Paris, France.
  Fact 0.888: Lines, Miss Mary Conover boarded at Cherbourg.
  Fact 0.889: Lines, Miss Mary Conover's destination is Hanover, New Hampshire, US.
  Fact 0.890: Lines, Miss Mary Conover has 9 in the lifeboat or body column.
  Fact 0.891: Long, Mr. Milton Clyde was 29 years old.
  Fact 0.892: Long, Mr. Milton Clyde is from Springfield, Massachusetts, US.
  Fact 0.893: Long, Mr. Milton Clyde boarded at Southampton.
  Fact 0.894: Long, Mr. Milton Clyde's destination is Springfield, Massachusetts, US.
  Fact 0.895: Long, Mr. Milton Clyde has 126 in the lifeboat or body column.
  Fact 0.896: Longley, Miss Gretchen Fiske was 21 years old.
  Fact 0.897: Longley, Miss Gretchen Fiske is from Hudson, New York, US.
  Fact 0.898: Longley, Miss Gretchen Fiske boarded at Cherbourg.
  Fact 0.899: Longley, Miss Gretchen Fiske's destination is Hudson, New York, US.
  Fact 0.900: Longley, Miss Gretchen Fiske has 10 in the lifeboat or body column.
  Fact 0.901: Loring, Mr. Joseph Holland was 30 years old.
  Fact 0.902: Loring, Mr. Joseph Holland is from London, England, UK.
  Fact 0.903: Loring, Mr. Joseph Holland boarded at Southampton.
  Fact 0.904: Loring, Mr. Joseph Holland's destination is New York City.
  Fact 0.905: Madill, Miss Georgette Alexandra was 16 years old.
  Fact 0.906: Madill, Miss Georgette Alexandra is from St. Louis, Missouri, US.
  Fact 0.907: Madill, Miss Georgette Alexandra boarded at Southampton.
  Fact 0.908: Madill, Miss Georgette Alexandra's destination is St. Louis, Missouri, US.
  Fact 0.909: Maguire, Mr. John Edward was 30 years old.
  Fact 0.910: Maguire, Mr. John Edward is from Brockton, Massachusetts, US.
  Fact 0.911: Maguire, Mr. John Edward boarded at Southampton.
  Fact 0.912: Maguire, Mr. John Edward's destination is Brockton, Massachusetts, US.
  Fact 0.913: Maréchal, Mr. Pierre, Sr. was 28 years old.
  Fact 0.914: Maréchal, Mr. Pierre, Sr. is from Paris, France.
  Fact 0.915: Maréchal, Mr. Pierre, Sr. boarded at Cherbourg.
  Fact 0.916: Maréchal, Mr. Pierre, Sr.'s destination is New York City.
  Fact 0.917: Maréchal, Mr. Pierre, Sr. has 7 in the lifeboat or body column.
  Fact 0.918: Marvin, Mr. Daniel Warner was 18 years old.
  Fact 0.919: Marvin, Mr. Daniel Warner is from New York City.
  Fact 0.920: Marvin, Mr. Daniel Warner boarded at Southampton.
  Fact 0.921: Marvin, Mr. Daniel Warner's destination is New York City.
  Fact 0.922: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) was 18 years old.
  Fact 0.923: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) is from New York City.
  Fact 0.924: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) boarded at Southampton.
  Fact 0.925: Marvin, Mrs. Mary Graham Carmichael (née Farquarson)'s destination is New York City.
  Fact 0.926: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) has 10 in the lifeboat or body column.
  Fact 0.927: Mayné, Miss Bertha Antonine was 24 years old.
  Fact 0.928: Mayné, Miss Bertha Antonine is from Brussels, Belgium.
  Fact 0.929: Mayné, Miss Bertha Antonine boarded at Cherbourg.
  Fact 0.930: Mayné, Miss Bertha Antonine's destination is Montreal, Quebec, Canada.
  Fact 0.931: Mayné, Miss Bertha Antonine has 6 in the lifeboat or body column.
  Fact 0.932: McCaffry, Mr. Thomas Francis was 46 years old.
  Fact 0.933: McCaffry, Mr. Thomas Francis is from Vancouver, British Columbia, Canada.
  Fact 0.934: McCaffry, Mr. Thomas Francis boarded at Southampton.
  Fact 0.935: McCaffry, Mr. Thomas Francis's destination is Vancouver, British Columbia, Canada.
  Fact 0.936: McCaffry, Mr. Thomas Francis has 292 in the lifeboat or body column.
  Fact 0.937: McCarthy, Mr. Timothy J. was 54 years old.
  Fact 0.938: McCarthy, Mr. Timothy J. is from Dorchester, Massachusetts, US.
  Fact 0.939: McCarthy, Mr. Timothy J. boarded at Southampton.
  Fact 0.940: McCarthy, Mr. Timothy J.'s destination is Dorchester, Massachusetts, US.
  Fact 0.941: McCarthy, Mr. Timothy J. has 175 in the lifeboat or body column.
  Fact 0.942: McGough, Mr. James Robert was 35 years old.
  Fact 0.943: McGough, Mr. James Robert is from Philadelphia, Pennsylvania, US.
  Fact 0.944: McGough, Mr. James Robert boarded at Southampton.
  Fact 0.945: McGough, Mr. James Robert's destination is Philadelphia, Pennsylvania, US.
  Fact 0.946: McGough, Mr. James Robert has 7 in the lifeboat or body column.
  Fact 0.947: Meyer, Mr. Edgar Joseph was 28 years old.
  Fact 0.948: Meyer, Mr. Edgar Joseph is from New York City.
  Fact 0.949: Meyer, Mr. Edgar Joseph boarded at Cherbourg.
  Fact 0.950: Meyer, Mr. Edgar Joseph's destination is New York City.
  Fact 0.951: Meyer, Mrs. Leila (née Saks) was 25 years old.
  Fact 0.952: Meyer, Mrs. Leila (née Saks) is from New York City.
  Fact 0.953: Meyer, Mrs. Leila (née Saks) boarded at Cherbourg.
  Fact 0.954: Meyer, Mrs. Leila (née Saks)'s destination is New York City.
  Fact 0.955: Meyer, Mrs. Leila (née Saks) has 6 in the lifeboat or body column.
  Fact 0.956: Millet, Mr. Francis Davis was 65 years old.
  Fact 0.957: Millet, Mr. Francis Davis is from East Bridgewater, Massachusetts, US.
  Fact 0.958: Millet, Mr. Francis Davis boarded at Cherbourg.
  Fact 0.959: Millet, Mr. Francis Davis's destination is East Bridgewater, Massachusetts, US.
  Fact 0.960: Millet, Mr. Francis Davis has 249 in the lifeboat or body column.
  Fact 0.961: Minahan, Dr. William Edward was 44 years old.
  Fact 0.962: Minahan, Dr. William Edward is from Fond du Lac, Wisconsin, US.
  Fact 0.963: Minahan, Dr. William Edward boarded at Southampton.
  Fact 0.964: Minahan, Dr. William Edward's destination is Fond du Lac, Wisconsin, US.
  Fact 0.965: Minahan, Dr. William Edward has 230 in the lifeboat or body column.
  Fact 0.966: Minahan, Mrs. Lillian E. (née Thorpe) was 37 years old.
  Fact 0.967: Minahan, Mrs. Lillian E. (née Thorpe) is from Fond du Lac, Wisconsin, US.
  Fact 0.968: Minahan, Mrs. Lillian E. (née Thorpe) boarded at Southampton.
  Fact 0.969: Minahan, Mrs. Lillian E. (née Thorpe)'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.970: Minahan, Mrs. Lillian E. (née Thorpe) has 14 in the lifeboat or body column.
  Fact 0.971: Minahan, Miss Daisy E. was 33 years old.
  Fact 0.972: Minahan, Miss Daisy E. is from Fond du Lac, Wisconsin, US.
  Fact 0.973: Minahan, Miss Daisy E. boarded at Southampton.
  Fact 0.974: Minahan, Miss Daisy E.'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.975: Minahan, Miss Daisy E. has 14 in the lifeboat or body column.
  Fact 0.976: Mock, Mr. Philipp Edmund was 30 years old.
  Fact 0.977: Mock, Mr. Philipp Edmund is from New York City.
  Fact 0.978: Mock, Mr. Philipp Edmund boarded at Cherbourg.
  Fact 0.979: Mock, Mr. Philipp Edmund's destination is New York City.
  Fact 0.980: Mock, Mr. Philipp Edmund has 11 in the lifeboat or body column.
  Fact 0.981: Molson, Mr. Harry Markland was 55 years old.
  Fact 0.982: Molson, Mr. Harry Markland is from Montreal, Quebec, Canada.
  Fact 0.983: Molson, Mr. Harry Markland boarded at Southampton.
  Fact 0.984: Molson, Mr. Harry Markland's destination is Montreal, Quebec, Canada.
  Fact 0.985: Moore, Mr. Clarence Bloomfield was 47 years old.
  Fact 0.986: Moore, Mr. Clarence Bloomfield is from Washington, D.C., US.
  Fact 0.987: Moore, Mr. Clarence Bloomfield boarded at Southampton.
  Fact 0.988: Moore, Mr. Clarence Bloomfield's destination is Washington, D.C., US.
  Fact 0.989: Mr. Charles Henry Harrington was 37 years old.
  Fact 0.990: Mr. Charles Henry Harrington is listed as the valet.
  Fact 0.991: Mr. Charles Henry Harrington is from Washington, D.C., US.
  Fact 0.992: Mr. Charles Henry Harrington boarded at Southampton.
  Fact 0.993: Mr. Charles Henry Harrington's destination is Washington, D.C., US.
  Fact 0.994: Natsch, Mr. Charles was 36 years old.
  Fact 0.995: Natsch, Mr. Charles is from Brooklyn, New York, US.
  Fact 0.996: Natsch, Mr. Charles boarded at Cherbourg.
  Fact 0.997: Natsch, Mr. Charles's destination is Brooklyn, New York, US.
  Fact 0.998: Newell, Mr. Arthur Webster was 58 years old.
  Fact 0.999: Newell, Mr. Arthur Webster is from Lexington, Massachusetts, US.
  Fact 0.1000: Newell, Mr. Arthur Webster boarded at Cherbourg.
  Fact 0.1001: Newell, Mr. Arthur Webster's destination is Lexington, Massachusetts, US.
  Fact 0.1002: Newell, Mr. Arthur Webster has 122 in the lifeboat or body column.
  Fact 0.1003: Newell, Miss Madeleine was 31 years old.
  Fact 0.1004: Newell, Miss Madeleine is from Lexington, Massachusetts, US.
  Fact 0.1005: Newell, Miss Madeleine boarded at Cherbourg.
  Fact 0.1006: Newell, Miss Madeleine's destination is Lexington, Massachusetts, US.
  Fact 0.1007: Newell, Miss Madeleine has 6 in the lifeboat or body column.
  Fact 0.1008: Newell, Miss Marjorie Anne was 23 years old.
  Fact 0.1009: Newell, Miss Marjorie Anne is from Lexington, Massachusetts, US.
  Fact 0.1010: Newell, Miss Marjorie Anne boarded at Cherbourg.
  Fact 0.1011: Newell, Miss Marjorie Anne's destination is Lexington, Massachusetts, US.
  Fact 0.1012: Newell, Miss Marjorie Anne has 6 in the lifeboat or body column.
  Fact 0.1013: Newsom, Miss Helen Monypeny was 19 years old.
  Fact 0.1014: Newsom, Miss Helen Monypeny is from New York City.
  Fact 0.1015: Newsom, Miss Helen Monypeny boarded at Southampton.
  Fact 0.1016: Newsom, Miss Helen Monypeny's destination is New York City.
  Fact 0.1017: Newsom, Miss Helen Monypeny has 5 in the lifeboat or body column.
  Fact 0.1018: Nicholson, Mr. Arthur Ernest was 64 years old.
  Fact 0.1019: Nicholson, Mr. Arthur Ernest is from Shanklin, Isle of Wight, England, UK.
  Fact 0.1020: Nicholson, Mr. Arthur Ernest boarded at Southampton.
  Fact 0.1021: Nicholson, Mr. Arthur Ernest's destination is New York City.
  Fact 0.1022: Nicholson, Mr. Arthur Ernest has 263 in the lifeboat or body column.
  Fact 0.1023: Nourney, Mr. Alfred was 20 years old.
  Fact 0.1024: Nourney, Mr. Alfred is from Cologne, German Empire.
  Fact 0.1025: Nourney, Mr. Alfred boarded at Cherbourg.
  Fact 0.1026: Nourney, Mr. Alfred's destination is New York City.
  Fact 0.1027: Nourney, Mr. Alfred has 7 in the lifeboat or body column.
  Fact 0.1028: Omont, Mr. Alfred Fernand was 29 years old.
  Fact 0.1029: Omont, Mr. Alfred Fernand is from Le Havre, France.
  Fact 0.1030: Omont, Mr. Alfred Fernand boarded at Cherbourg.
  Fact 0.1031: Omont, Mr. Alfred Fernand's destination is New York City.
  Fact 0.1032: Omont, Mr. Alfred Fernand has 7 in the lifeboat or body column.
  Fact 0.1033: Ostby, Mr. Engelhart Cornelius was 64 years old.
  Fact 0.1034: Ostby, Mr. Engelhart Cornelius is from Providence, Rhode Island, US.
  Fact 0.1035: Ostby, Mr. Engelhart Cornelius boarded at Southampton.
  Fact 0.1036: Ostby, Mr. Engelhart Cornelius's destination is Providence, Rhode Island, US.
  Fact 0.1037: Ostby, Mr. Engelhart Cornelius has 234 in the lifeboat or body column.
  Fact 0.1038: Ostby, Miss Helene Ragnhild was 22 years old.
  Fact 0.1039: Ostby, Miss Helene Ragnhild is from Providence, Rhode Island, US.
  Fact 0.1040: Ostby, Miss Helene Ragnhild boarded at Southampton.
  Fact 0.1041: Ostby, Miss Helene Ragnhild's destination is Providence, Rhode Island, US.
  Fact 0.1042: Ostby, Miss Helene Ragnhild has 5 in the lifeboat or body column.
  Fact 0.1043: Ovies y Rodriguez, Mr. Servando José Florentino was 36 years old.
  Fact 0.1044: Ovies y Rodriguez, Mr. Servando José Florentino is from Havana, Cuba.
  Fact 0.1045: Ovies y Rodriguez, Mr. Servando José Florentino boarded at Cherbourg.
  Fact 0.1046: Ovies y Rodriguez, Mr. Servando José Florentino's destination is Havana, Cuba.
  Fact 0.1047: Ovies y Rodriguez, Mr. Servando José Florentino has 189 in the lifeboat or body column.
  Fact 0.1048: Parr, Mr. William Henry Marsh was 29 years old.
  Fact 0.1049: Parr, Mr. William Henry Marsh is from Belfast, Ireland, UK.
  Fact 0.1050: Parr, Mr. William Henry Marsh boarded at Belfast.
  Fact 0.1051: Parr, Mr. William Henry Marsh's destination is New York City.
  Fact 0.1052: Partner, Mr. Austin was 40 years old.
  Fact 0.1053: Partner, Mr. Austin is from Tolworth, London, England, UK.
  Fact 0.1054: Partner, Mr. Austin boarded at Southampton.
  Fact 0.1055: Partner, Mr. Austin's destination is Toronto, Ontario, Canada.
  Fact 0.1056: Partner, Mr. Austin has 166 in the lifeboat or body column.
  Fact 0.1057: Pears, Mr. Thomas Clinton was 29 years old.
  Fact 0.1058: Pears, Mr. Thomas Clinton is from Isleworth, London, England, UK.
  Fact 0.1059: Pears, Mr. Thomas Clinton boarded at Southampton.
  Fact 0.1060: Pears, Mr. Thomas Clinton's destination is New York City.
  Fact 0.1061: Pears, Mrs. Edith (née Wearne) was 22 years old.
  Fact 0.1062: Pears, Mrs. Edith (née Wearne) is from Isleworth, London, England, UK.
  Fact 0.1063: Pears, Mrs. Edith (née Wearne) boarded at Southampton.
  Fact 0.1064: Pears, Mrs. Edith (née Wearne)'s destination is New York City.
  Fact 0.1065: Pears, Mrs. Edith (née Wearne) has 8 in the lifeboat or body column.
  Fact 0.1066: Peñasco y Castellana, Mr. Victor was 24 years old.
  Fact 0.1067: Peñasco y Castellana, Mr. Victor is from Madrid, Spain.
  Fact 0.1068: Peñasco y Castellana, Mr. Victor boarded at Cherbourg.
  Fact 0.1069: Peñasco y Castellana, Mr. Victor's destination is New York City.
  Fact 0.1070: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) was 22 years old.
  Fact 0.1071: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) is from Madrid, Spain.
  Fact 0.1072: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) boarded at Cherbourg.
  Fact 0.1073: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo)'s destination is New York City.
  Fact 0.1074: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) has 8 in the lifeboat or body column.
  Fact 0.1075: Doña Fermina Oliva y Ocana was 39 years old.
  Fact 0.1076: Doña Fermina Oliva y Ocana is listed as the maid.
  Fact 0.1077: Doña Fermina Oliva y Ocana is from Madrid, Spain.
  Fact 0.1078: Doña Fermina Oliva y Ocana boarded at Cherbourg.
  Fact 0.1079: Doña Fermina Oliva y Ocana's destination is New York City.
  Fact 0.1080: Doña Fermina Oliva y Ocana has 8 in the lifeboat or body column.
  Fact 0.1081: Peuchen, Major Arthur Godfrey was 52 years old.
  Fact 0.1082: Peuchen, Major Arthur Godfrey is from Toronto, Ontario, Canada.
  Fact 0.1083: Peuchen, Major Arthur Godfrey boarded at Southampton.
  Fact 0.1084: Peuchen, Major Arthur Godfrey's destination is Toronto, Ontario, Canada.
  Fact 0.1085: Peuchen, Major Arthur Godfrey has 6 in the lifeboat or body column.
  Fact 0.1086: Porter, Mr. Walter Chamberlain was 46 years old.
  Fact 0.1087: Porter, Mr. Walter Chamberlain is from Worcester, Massachusetts, US.
  Fact 0.1088: Porter, Mr. Walter Chamberlain boarded at Southampton.
  Fact 0.1089: Porter, Mr. Walter Chamberlain's destination is Worcester, Massachusetts, US.
  Fact 0.1090: Porter, Mr. Walter Chamberlain has 207 in the lifeboat or body column.
  Fact 0.1091: Potter, Mrs. Lily Alexenia (née Wilson) was 56 years old.
  Fact 0.1092: Potter, Mrs. Lily Alexenia (née Wilson) is from Philadelphia, Pennsylvania, US.
  Fact 0.1093: Potter, Mrs. Lily Alexenia (née Wilson) boarded at Cherbourg.
  Fact 0.1094: Potter, Mrs. Lily Alexenia (née Wilson)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1095: Potter, Mrs. Lily Alexenia (née Wilson) has 7 in the lifeboat or body column.
  Fact 0.1096: Reuchlin, The Honourable Mr. Johan George was 38 years old.
  Fact 0.1097: Reuchlin, The Honourable Mr. Johan George is from Rotterdam, The Netherlands.
  Fact 0.1098: Reuchlin, The Honourable Mr. Johan George boarded at Southampton.
  Fact 0.1099: Reuchlin, The Honourable Mr. Johan George's destination is New York City.
  Fact 0.1100: Rheims, Mr. George Alexander Lucien was 36 years old.
  Fact 0.1101: Rheims, Mr. George Alexander Lucien is from Paris, France.
  Fact 0.1102: Rheims, Mr. George Alexander Lucien boarded at Southampton.
  Fact 0.1103: Rheims, Mr. George Alexander Lucien's destination is New York City.
  Fact 0.1104: Robert, Mrs. Elisabeth Walton (née McMillan) was 43 years old.
  Fact 0.1105: Robert, Mrs. Elisabeth Walton (née McMillan) is from St. Louis, Missouri, US.
  Fact 0.1106: Robert, Mrs. Elisabeth Walton (née McMillan) boarded at Southampton.
  Fact 0.1107: Robert, Mrs. Elisabeth Walton (née McMillan)'s destination is St. Louis, Missouri, US.
  Fact 0.1108: Miss Emilie Kreuchen was 29 years old.
  Fact 0.1109: Miss Emilie Kreuchen is listed as the maid.
  Fact 0.1110: Miss Emilie Kreuchen is from St. Louis, Missouri, US.
  Fact 0.1111: Miss Emilie Kreuchen boarded at Southampton.
  Fact 0.1112: Miss Emilie Kreuchen's destination is St. Louis, Missouri, US.
  Fact 0.1113: Roebling, Mr. Washington Augustus II was 31 years old.
  Fact 0.1114: Roebling, Mr. Washington Augustus II is from Trenton, New Jersey, US.
  Fact 0.1115: Roebling, Mr. Washington Augustus II boarded at Southampton.
  Fact 0.1116: Roebling, Mr. Washington Augustus II's destination is Trenton, New Jersey, US.
  Fact 0.1117: Romaine, Mr. Charles Hallance (alias C. Rolmane) was 45 years old.
  Fact 0.1118: Romaine, Mr. Charles Hallance (alias C. Rolmane) is from New York City.
  Fact 0.1119: Romaine, Mr. Charles Hallance (alias C. Rolmane) boarded at Southampton.
  Fact 0.1120: Romaine, Mr. Charles Hallance (alias C. Rolmane)'s destination is New York City.
  Fact 0.1121: Romaine, Mr. Charles Hallance (alias C. Rolmane) has 9 in the lifeboat or body column.
  Fact 0.1122: Rood, Mr. Hugh Roscoe was 38 years old.
  Fact 0.1123: Rood, Mr. Hugh Roscoe is from Seattle, Washington, US.
  Fact 0.1124: Rood, Mr. Hugh Roscoe boarded at Southampton.
  Fact 0.1125: Rood, Mr. Hugh Roscoe's destination is Seattle, Washington, US.
  Fact 0.1126: Rosenbaum, Miss Edith Louise was 34 years old.
  Fact 0.1127: Rosenbaum, Miss Edith Louise is from Paris, France.
  Fact 0.1128: Rosenbaum, Miss Edith Louise boarded at Cherbourg.
  Fact 0.1129: Rosenbaum, Miss Edith Louise's destination is Cincinnati, Ohio, US.
  Fact 0.1130: Rosenbaum, Miss Edith Louise has 11 in the lifeboat or body column.
  Fact 0.1131: Rosenshine, Mr. George (alias George Thorne) was 46 years old.
  Fact 0.1132: Rosenshine, Mr. George (alias George Thorne) is from New York City.
  Fact 0.1133: Rosenshine, Mr. George (alias George Thorne) boarded at Cherbourg.
  Fact 0.1134: Rosenshine, Mr. George (alias George Thorne)'s destination is New York City.
  Fact 0.1135: Rosenshine, Mr. George (alias George Thorne) has 16 in the lifeboat or body column.
  Fact 0.1136: Ross, Mr. John Hugo was 36 years old.
  Fact 0.1137: Ross, Mr. John Hugo is from Winnipeg, Manitoba, Canada.
  Fact 0.1138: Ross, Mr. John Hugo boarded at Southampton.
  Fact 0.1139: Ross, Mr. John Hugo's destination is Winnipeg, Manitoba, Canada.
  Fact 0.1140: Rothschild, Mr. Martin was 46 years old.
  Fact 0.1141: Rothschild, Mr. Martin is from New York City.
  Fact 0.1142: Rothschild, Mr. Martin boarded at Cherbourg.
  Fact 0.1143: Rothschild, Mr. Martin's destination is New York City.
  Fact 0.1144: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) was 54 years old.
  Fact 0.1145: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) is from New York City.
  Fact 0.1146: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) boarded at Cherbourg.
  Fact 0.1147: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett)'s destination is New York City.
  Fact 0.1148: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) has 6 in the lifeboat or body column.
  Fact 0.1149: Rowe, Mr. Alfred G. was 59 years old.
  Fact 0.1150: Rowe, Mr. Alfred G. is from Liverpool, Merseyside, England, UK.
  Fact 0.1151: Rowe, Mr. Alfred G. boarded at Southampton.
  Fact 0.1152: Rowe, Mr. Alfred G.'s destination is New York City.
  Fact 0.1153: Rowe, Mr. Alfred G. has 109 in the lifeboat or body column.
  Fact 0.1154: Ryerson, Mr. Arthur Larned was 61 years old.
  Fact 0.1155: Ryerson, Mr. Arthur Larned is from Cooperstown, New York, US.
  Fact 0.1156: Ryerson, Mr. Arthur Larned boarded at Cherbourg.
  Fact 0.1157: Ryerson, Mr. Arthur Larned's destination is Cooperstown, New York, US.
  Fact 0.1158: Ryerson, Mrs. Emily Maria (née Borie) was 48 years old.
  Fact 0.1159: Ryerson, Mrs. Emily Maria (née Borie) is from Cooperstown, New York, US.
  Fact 0.1160: Ryerson, Mrs. Emily Maria (née Borie) boarded at Cherbourg.
  Fact 0.1161: Ryerson, Mrs. Emily Maria (née Borie)'s destination is Cooperstown, New York, US.
  Fact 0.1162: Miss Victorine Chaudanson was 36 years old.
  Fact 0.1163: Miss Victorine Chaudanson is listed as the maid.
  Fact 0.1164: Miss Victorine Chaudanson is from Cooperstown, New York, US.
  Fact 0.1165: Miss Victorine Chaudanson boarded at Cherbourg.
  Fact 0.1166: Miss Victorine Chaudanson's destination is Cooperstown, New York, US.
  Fact 0.1167: Ryerson, Miss Susan Parker "Suzette" was 21 years old.
  Fact 0.1168: Ryerson, Miss Susan Parker "Suzette" is from Cooperstown, New York, US.
  Fact 0.1169: Ryerson, Miss Susan Parker "Suzette" boarded at Cherbourg.
  Fact 0.1170: Ryerson, Miss Susan Parker "Suzette"'s destination is Cooperstown, New York, US.
  Fact 0.1171: Ryerson, Miss Emily Borie was 18 years old.
  Fact 0.1172: Ryerson, Miss Emily Borie is from Cooperstown, New York, US.
  Fact 0.1173: Ryerson, Miss Emily Borie boarded at Cherbourg.
  Fact 0.1174: Ryerson, Miss Emily Borie's destination is Cooperstown, New York, US.
  Fact 0.1175: Ryerson, Master John Borie "Jack" was 13 years old.
  Fact 0.1176: Ryerson, Master John Borie "Jack" is from Cooperstown, New York, US.
  Fact 0.1177: Ryerson, Master John Borie "Jack" boarded at Cherbourg.
  Fact 0.1178: Ryerson, Master John Borie "Jack"'s destination is Cooperstown, New York, US.
  Fact 0.1179: Miss Grace Scott Bowen was 45 years old.
  Fact 0.1180: Miss Grace Scott Bowen is listed as the governess.
  Fact 0.1181: Miss Grace Scott Bowen is from Cooperstown, New York, US.
  Fact 0.1182: Miss Grace Scott Bowen boarded at Cherbourg.
  Fact 0.1183: Miss Grace Scott Bowen's destination is Cooperstown, New York, US.
  Fact 0.1184: Saalfeld, Mr. Adolphe was 47 years old.
  Fact 0.1185: Saalfeld, Mr. Adolphe is from Manchester, England, UK.
  Fact 0.1186: Saalfeld, Mr. Adolphe boarded at Southampton.
  Fact 0.1187: Saalfeld, Mr. Adolphe's destination is New York City.
  Fact 0.1188: Salomon, Mr. Abraham Lincoln was 43 years old.
  Fact 0.1189: Salomon, Mr. Abraham Lincoln is from New York City.
  Fact 0.1190: Salomon, Mr. Abraham Lincoln boarded at Cherbourg.
  Fact 0.1191: Salomon, Mr. Abraham Lincoln's destination is New York City.
  Fact 0.1192: Schabert, Mrs. Emma (née Mock) was 35 years old.
  Fact 0.1193: Schabert, Mrs. Emma (née Mock) is from Hamburg, German Empire.
  Fact 0.1194: Schabert, Mrs. Emma (née Mock) boarded at Southampton.
  Fact 0.1195: Schabert, Mrs. Emma (née Mock)'s destination is New York City.
  Fact 0.1196: Schabert, Mrs. Emma (née Mock) has 11 in the lifeboat or body column.
  Fact 0.1197: Seward, Mr. Frederic Kimber was 34 years old.
  Fact 0.1198: Seward, Mr. Frederic Kimber is from New York City.
  Fact 0.1199: Seward, Mr. Frederic Kimber boarded at Southampton.
  Fact 0.1200: Seward, Mr. Frederic Kimber's destination is New York City.
  Fact 0.1201: Seward, Mr. Frederic Kimber has 7 in the lifeboat or body column.
  Fact 0.1202: Silverthorne, Mr. Spencer Victor was 35 years old.
  Fact 0.1203: Silverthorne, Mr. Spencer Victor is from St. Louis, Missouri, US.
  Fact 0.1204: Silverthorne, Mr. Spencer Victor boarded at Southampton.
  Fact 0.1205: Silverthorne, Mr. Spencer Victor's destination is St. Louis, Missouri, US.
  Fact 0.1206: Silverthorne, Mr. Spencer Victor has 5 in the lifeboat or body column.
  Fact 0.1207: Silvey, Mr. William Baird was 50 years old.
  Fact 0.1208: Silvey, Mr. William Baird is from Duluth, Minnesota, US.
  Fact 0.1209: Silvey, Mr. William Baird boarded at Cherbourg.
  Fact 0.1210: Silvey, Mr. William Baird's destination is Duluth, Minnesota, US.
  Fact 0.1211: Silvey, Mrs. Alice (née Munger) was 39 years old.
  Fact 0.1212: Silvey, Mrs. Alice (née Munger) is from Duluth, Minnesota, US.
  Fact 0.1213: Silvey, Mrs. Alice (née Munger) boarded at Cherbourg.
  Fact 0.1214: Silvey, Mrs. Alice (née Munger)'s destination is Duluth, Minnesota, US.
  Fact 0.1215: Silvey, Mrs. Alice (née Munger) has 11 in the lifeboat or body column.
  Fact 0.1216: Simonius-Blumer, Colonel Alfons was 56 years old.
  Fact 0.1217: Simonius-Blumer, Colonel Alfons is from Basel, Switzerland.
  Fact 0.1218: Simonius-Blumer, Colonel Alfons boarded at Southampton.
  Fact 0.1219: Simonius-Blumer, Colonel Alfons's destination is New York City.
  Fact 0.1220: Sloper, Mr. William Thompson was 28 years old.
  Fact 0.1221: Sloper, Mr. William Thompson is from New Britain, Connecticut, US.
  Fact 0.1222: Sloper, Mr. William Thompson boarded at Southampton.
  Fact 0.1223: Sloper, Mr. William Thompson's destination is New Britain, Connecticut, US.
  Fact 0.1224: Sloper, Mr. William Thompson has 7 in the lifeboat or body column.
  Fact 0.1225: Smart, Mr. John Montgomery was 56 years old.
  Fact 0.1226: Smart, Mr. John Montgomery is from Kildale, North Yorkshire, England, UK.
  Fact 0.1227: Smart, Mr. John Montgomery boarded at Southampton.
  Fact 0.1228: Smart, Mr. John Montgomery's destination is New York City.
  Fact 0.1229: Smith, Mr. James Clinch was 56 years old.
  Fact 0.1230: Smith, Mr. James Clinch is from Paris, France.
  Fact 0.1231: Smith, Mr. James Clinch boarded at Cherbourg.
  Fact 0.1232: Smith, Mr. James Clinch's destination is Long Island, New York, US.
  Fact 0.1233: Smith, Mr. Lucien Philip was 24 years old.
  Fact 0.1234: Smith, Mr. Lucien Philip is from Huntington, West Virginia, US.
  Fact 0.1235: Smith, Mr. Lucien Philip boarded at Cherbourg.
  Fact 0.1236: Smith, Mr. Lucien Philip's destination is Huntington, West Virginia, US.
  Fact 0.1237: Smith, Mrs. Mary Eloise (née Hughes) was 18 years old.
  Fact 0.1238: Smith, Mrs. Mary Eloise (née Hughes) is from Huntington, West Virginia, US.
  Fact 0.1239: Smith, Mrs. Mary Eloise (née Hughes) boarded at Cherbourg.
  Fact 0.1240: Smith, Mrs. Mary Eloise (née Hughes)'s destination is Huntington, West Virginia, US.
  Fact 0.1241: Smith, Mrs. Mary Eloise (née Hughes) has 6 in the lifeboat or body column.
  Fact 0.1242: Smith, Mr. Richard William was 57 years old.
  Fact 0.1243: Smith, Mr. Richard William is from Streatham, London, England, UK.
  Fact 0.1244: Smith, Mr. Richard William boarded at Southampton.
  Fact 0.1245: Smith, Mr. Richard William's destination is New York City.
  Fact 0.1246: Snyder, Mr. John Pillsbury was 24 years old.
  Fact 0.1247: Snyder, Mr. John Pillsbury is from Minneapolis, Minnesota, US.
  Fact 0.1248: Snyder, Mr. John Pillsbury boarded at Southampton.
  Fact 0.1249: Snyder, Mr. John Pillsbury's destination is Minneapolis, Minnesota, US.
  Fact 0.1250: Snyder, Mr. John Pillsbury has 7 in the lifeboat or body column.
  Fact 0.1251: Snyder, Mrs. Nellie (née Stevenson) was 23 years old.
  Fact 0.1252: Snyder, Mrs. Nellie (née Stevenson) is from Minneapolis, Minnesota, US.
  Fact 0.1253: Snyder, Mrs. Nellie (née Stevenson) boarded at Southampton.
  Fact 0.1254: Snyder, Mrs. Nellie (née Stevenson)'s destination is Minneapolis, Minnesota, US.
  Fact 0.1255: Snyder, Mrs. Nellie (née Stevenson) has 7 in the lifeboat or body column.
  Fact 0.1256: Spedden, Mr. Frederic Oakley was 45 years old.
  Fact 0.1257: Spedden, Mr. Frederic Oakley is from Tuxedo Park, New York, US.
  Fact 0.1258: Spedden, Mr. Frederic Oakley boarded at Cherbourg.
  Fact 0.1259: Spedden, Mr. Frederic Oakley's destination is Tuxedo Park, New York, US.
  Fact 0.1260: Spedden, Mrs. Margaretta Corning (née Stone) was 39 years old.
  Fact 0.1261: Spedden, Mrs. Margaretta Corning (née Stone) is from Tuxedo Park, New York, US.
  Fact 0.1262: Spedden, Mrs. Margaretta Corning (née Stone) boarded at Cherbourg.
  Fact 0.1263: Spedden, Mrs. Margaretta Corning (née Stone)'s destination is Tuxedo Park, New York, US.
  Fact 0.1264: Miss Helen Alice Wilson was 31 years old.
  Fact 0.1265: Miss Helen Alice Wilson is listed as the maid.
  Fact 0.1266: Miss Helen Alice Wilson is from Tuxedo Park, New York, US.
  Fact 0.1267: Miss Helen Alice Wilson boarded at Cherbourg.
  Fact 0.1268: Miss Helen Alice Wilson's destination is Tuxedo Park, New York, US.
  Fact 0.1269: Spedden, Master Robert Douglas was 6 years old.
  Fact 0.1270: Spedden, Master Robert Douglas is from Tuxedo Park, New York, US.
  Fact 0.1271: Spedden, Master Robert Douglas boarded at Cherbourg.
  Fact 0.1272: Spedden, Master Robert Douglas's destination is Tuxedo Park, New York, US.
  Fact 0.1273: Miss Elizabeth Margaret Burns was 41 years old.
  Fact 0.1274: Miss Elizabeth Margaret Burns is listed as the nurse.
  Fact 0.1275: Miss Elizabeth Margaret Burns is from Tuxedo Park, New York, US.
  Fact 0.1276: Miss Elizabeth Margaret Burns boarded at Cherbourg.
  Fact 0.1277: Miss Elizabeth Margaret Burns's destination is Tuxedo Park, New York, US.
  Fact 0.1278: Spencer, Mr. William Augustus was 57 years old.
  Fact 0.1279: Spencer, Mr. William Augustus is from New York City.
  Fact 0.1280: Spencer, Mr. William Augustus boarded at Cherbourg.
  Fact 0.1281: Spencer, Mr. William Augustus's destination is New York City.
  Fact 0.1282: Spencer, Mrs. Marie Eugénie (née Demougeot) was 45 years old.
  Fact 0.1283: Spencer, Mrs. Marie Eugénie (née Demougeot) is from New York City.
  Fact 0.1284: Spencer, Mrs. Marie Eugénie (née Demougeot) boarded at Cherbourg.
  Fact 0.1285: Spencer, Mrs. Marie Eugénie (née Demougeot)'s destination is New York City.
  Fact 0.1286: Spencer, Mrs. Marie Eugénie (née Demougeot) has 6 in the lifeboat or body column.
  Fact 0.1287: Miss Eugenie Elise Lurette was 59 years old.
  Fact 0.1288: Miss Eugenie Elise Lurette is listed as the maid.
  Fact 0.1289: Miss Eugenie Elise Lurette is from New York City.
  Fact 0.1290: Miss Eugenie Elise Lurette boarded at Cherbourg.
  Fact 0.1291: Miss Eugenie Elise Lurette's destination is New York City.
  Fact 0.1292: Miss Eugenie Elise Lurette has 6 in the lifeboat or body column.
  Fact 0.1293: Stähelin-Maeglin, Dr. Max was 32 years old.
  Fact 0.1294: Stähelin-Maeglin, Dr. Max is from Basel, Switzerland.
  Fact 0.1295: Stähelin-Maeglin, Dr. Max boarded at Southampton.
  Fact 0.1296: Stähelin-Maeglin, Dr. Max's destination is New York City.
  Fact 0.1297: Stead, Mr. William Thomas was 62 years old.
  Fact 0.1298: Stead, Mr. William Thomas is from London, England, UK.
  Fact 0.1299: Stead, Mr. William Thomas boarded at Southampton.
  Fact 0.1300: Stead, Mr. William Thomas's destination is New York City.
  Fact 0.1301: Stengel, Mr. Charles Emil Henry was 54 years old.
  Fact 0.1302: Stengel, Mr. Charles Emil Henry is from Newark, New Jersey, US.
  Fact 0.1303: Stengel, Mr. Charles Emil Henry boarded at Cherbourg.
  Fact 0.1304: Stengel, Mr. Charles Emil Henry's destination is Newark, New Jersey, US.
  Fact 0.1305: Stengel, Mrs. Annie May (née Morris) was 44 years old.
  Fact 0.1306: Stengel, Mrs. Annie May (née Morris) is from Newark, New Jersey, US.
  Fact 0.1307: Stengel, Mrs. Annie May (née Morris) boarded at Cherbourg.
  Fact 0.1308: Stengel, Mrs. Annie May (née Morris)'s destination is Newark, New Jersey, US.
  Fact 0.1309: Stengel, Mrs. Annie May (née Morris) has 5 in the lifeboat or body column.
  Fact 0.1310: Stephenson, Mrs. Martha (née Eustis) was 52 years old.
  Fact 0.1311: Stephenson, Mrs. Martha (née Eustis) is from Paris, France.
  Fact 0.1312: Stephenson, Mrs. Martha (née Eustis) boarded at Cherbourg.
  Fact 0.1313: Stephenson, Mrs. Martha (née Eustis)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1314: Stewart, Mr. Albert A. was 54 years old.
  Fact 0.1315: Stewart, Mr. Albert A. is from Cincinnati, Ohio, US.
  Fact 0.1316: Stewart, Mr. Albert A. boarded at Cherbourg.
  Fact 0.1317: Stewart, Mr. Albert A.'s destination is Gallipolis, Ohio, US.
  Fact 0.1318: Stone, Mrs. Martha Evelyn (née Stevens) was 62 years old.
  Fact 0.1319: Stone, Mrs. Martha Evelyn (née Stevens) is from New York City.
  Fact 0.1320: Stone, Mrs. Martha Evelyn (née Stevens) boarded at Southampton.
  Fact 0.1321: Stone, Mrs. Martha Evelyn (née Stevens)'s destination is New York City.
  Fact 0.1322: Stone, Mrs. Martha Evelyn (née Stevens) has 6 in the lifeboat or body column.
  Fact 0.1323: Miss Amelie "Amelia" Icard was 38 years old.
  Fact 0.1324: Miss Amelie "Amelia" Icard is listed as the maid.
  Fact 0.1325: Miss Amelie "Amelia" Icard is from New York City.
  Fact 0.1326: Miss Amelie "Amelia" Icard boarded at Southampton.
  Fact 0.1327: Miss Amelie "Amelia" Icard's destination is New York City.
  Fact 0.1328: Miss Amelie "Amelia" Icard has 6 in the lifeboat or body column.
  Fact 0.1329: Straus, Mr. Isidor was 67 years old.
  Fact 0.1330: Straus, Mr. Isidor is from New York City.
  Fact 0.1331: Straus, Mr. Isidor boarded at Southampton.
  Fact 0.1332: Straus, Mr. Isidor's destination is New York City.
  Fact 0.1333: Straus, Mr. Isidor has 96 in the lifeboat or body column.
  Fact 0.1334: Mr. John Farthing was 57 years old.
  Fact 0.1335: Mr. John Farthing is listed as the valet.
  Fact 0.1336: Mr. John Farthing is from New York City.
  Fact 0.1337: Mr. John Farthing boarded at Southampton.
  Fact 0.1338: Mr. John Farthing's destination is New York City.
  Fact 0.1339: Straus, Mrs. Rosalie Ida (née Blun) was 63 years old.
  Fact 0.1340: Straus, Mrs. Rosalie Ida (née Blun) is from New York City.
  Fact 0.1341: Straus, Mrs. Rosalie Ida (née Blun) boarded at Southampton.
  Fact 0.1342: Straus, Mrs. Rosalie Ida (née Blun)'s destination is New York City.
  Fact 0.1343: Miss Ellen Bird was 31 years old.
  Fact 0.1344: Miss Ellen Bird is listed as the maid.
  Fact 0.1345: Miss Ellen Bird is from New York City.
  Fact 0.1346: Miss Ellen Bird boarded at Southampton.
  Fact 0.1347: Miss Ellen Bird's destination is New York City.
  Fact 0.1348: Miss Ellen Bird has 8 in the lifeboat or body column.
  Fact 0.1349: Sutton, Mr. Frederick was 61 years old.
  Fact 0.1350: Sutton, Mr. Frederick is from Haddonfield, New Jersey, US.
  Fact 0.1351: Sutton, Mr. Frederick boarded at Southampton.
  Fact 0.1352: Sutton, Mr. Frederick's destination is Haddonfield, New Jersey, US.
  Fact 0.1353: Sutton, Mr. Frederick has 46 in the lifeboat or body column.
  Fact 0.1354: Swift, Mrs. Margaret Welles (née Barron) was 46 years old.
  Fact 0.1355: Swift, Mrs. Margaret Welles (née Barron) is from New York City.
  Fact 0.1356: Swift, Mrs. Margaret Welles (née Barron) boarded at Southampton.
  Fact 0.1357: Swift, Mrs. Margaret Welles (née Barron)'s destination is New York City.
  Fact 0.1358: Swift, Mrs. Margaret Welles (née Barron) has 8 in the lifeboat or body column.
  Fact 0.1359: Taussig, Mr. Emil was 52 years old.
  Fact 0.1360: Taussig, Mr. Emil is from New York City.
  Fact 0.1361: Taussig, Mr. Emil boarded at Southampton.
  Fact 0.1362: Taussig, Mr. Emil's destination is New York City.
  Fact 0.1363: Taussig, Mrs. Tillie (née Mandelbaum) was 39 years old.
  Fact 0.1364: Taussig, Mrs. Tillie (née Mandelbaum) is from New York City.
  Fact 0.1365: Taussig, Mrs. Tillie (née Mandelbaum) boarded at Southampton.
  Fact 0.1366: Taussig, Mrs. Tillie (née Mandelbaum)'s destination is New York City.
  Fact 0.1367: Taussig, Mrs. Tillie (née Mandelbaum) has 8 in the lifeboat or body column.
  Fact 0.1368: Taussig, Miss Ruth was 18 years old.
  Fact 0.1369: Taussig, Miss Ruth is from New York City.
  Fact 0.1370: Taussig, Miss Ruth boarded at Southampton.
  Fact 0.1371: Taussig, Miss Ruth's destination is New York City.
  Fact 0.1372: Taussig, Miss Ruth has 8 in the lifeboat or body column.
  Fact 0.1373: Taylor, Mr. Elmer Zebley was 48 years old.
  Fact 0.1374: Taylor, Mr. Elmer Zebley is from London, England, UK.
  Fact 0.1375: Taylor, Mr. Elmer Zebley boarded at Southampton.
  Fact 0.1376: Taylor, Mr. Elmer Zebley's destination is East Orange, New Jersey, US.
  Fact 0.1377: Taylor, Mr. Elmer Zebley has 5 in the lifeboat or body column.
  Fact 0.1378: Taylor, Mrs. Juliet Cummins (née Wright) was 48 years old.
  Fact 0.1379: Taylor, Mrs. Juliet Cummins (née Wright) is from London, England, UK.
  Fact 0.1380: Taylor, Mrs. Juliet Cummins (née Wright) boarded at Southampton.
  Fact 0.1381: Taylor, Mrs. Juliet Cummins (née Wright)'s destination is East Orange, New Jersey, US.
  Fact 0.1382: Taylor, Mrs. Juliet Cummins (née Wright) has 5 in the lifeboat or body column.
  Fact 0.1383: Thayer, Mr. John Borland II was 49 years old.
  Fact 0.1384: Thayer, Mr. John Borland II is from Haverford, Pennsylvania, US.
  Fact 0.1385: Thayer, Mr. John Borland II boarded at Cherbourg.
  Fact 0.1386: Thayer, Mr. John Borland II's destination is Haverford, Pennsylvania, US.
  Fact 0.1387: Thayer, Mrs. Marian Longsteth (née Morris) was 39 years old.
  Fact 0.1388: Thayer, Mrs. Marian Longsteth (née Morris) is from Haverford, Pennsylvania, US.
  Fact 0.1389: Thayer, Mrs. Marian Longsteth (née Morris) boarded at Cherbourg.
  Fact 0.1390: Thayer, Mrs. Marian Longsteth (née Morris)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1391: Miss Margaret Fleming was 42 years old.
  Fact 0.1392: Miss Margaret Fleming is listed as the maid.
  Fact 0.1393: Miss Margaret Fleming is from Haverford, Pennsylvania, US.
  Fact 0.1394: Miss Margaret Fleming boarded at Cherbourg.
  Fact 0.1395: Miss Margaret Fleming's destination is Haverford, Pennsylvania, US.
  Fact 0.1396: Thayer, Mr. John Borland "Jack" III was 17 years old.
  Fact 0.1397: Thayer, Mr. John Borland "Jack" III is from Haverford, Pennsylvania, US.
  Fact 0.1398: Thayer, Mr. John Borland "Jack" III boarded at Cherbourg.
  Fact 0.1399: Thayer, Mr. John Borland "Jack" III's destination is Haverford, Pennsylvania, US.
  Fact 0.1400: Thorne, Mrs. Gertrude Maybelle (née McMinn) was 37 years old.
  Fact 0.1401: Thorne, Mrs. Gertrude Maybelle (née McMinn) is from New York City, New York, US.
  Fact 0.1402: Thorne, Mrs. Gertrude Maybelle (née McMinn) boarded at Cherbourg.
  Fact 0.1403: Thorne, Mrs. Gertrude Maybelle (née McMinn)'s destination is New York City, New York, US.
  Fact 0.1404: Tucker, Mr. Gilbert Milligan Jr. was 31 years old.
  Fact 0.1405: Tucker, Mr. Gilbert Milligan Jr. is from Albany, New York, US.
  Fact 0.1406: Tucker, Mr. Gilbert Milligan Jr. boarded at Cherbourg.
  Fact 0.1407: Tucker, Mr. Gilbert Milligan Jr.'s destination is Albany, New York, US.
  Fact 0.1408: Tucker, Mr. Gilbert Milligan Jr. has 7 in the lifeboat or body column.
  Fact 0.1409: Uruchurtu, Don Manuel E. was 40 years old.
  Fact 0.1410: Uruchurtu, Don Manuel E. is from Mexico City, Mexico.
  Fact 0.1411: Uruchurtu, Don Manuel E. boarded at Cherbourg.
  Fact 0.1412: Uruchurtu, Don Manuel E.'s destination is Mexico City, Mexico.
  Fact 0.1413: Van der Hoef, Mr. Wyckoff was 61 years old.
  Fact 0.1414: Van der Hoef, Mr. Wyckoff is from Brooklyn, New York, US.
  Fact 0.1415: Van der Hoef, Mr. Wyckoff boarded at Belfast.
  Fact 0.1416: Van der Hoef, Mr. Wyckoff's destination is Brooklyn, New York, US.
  Fact 0.1417: Van der Hoef, Mr. Wyckoff has 245 in the lifeboat or body column.
  Fact 0.1418: Walker, Mr. William Anderson was 48 years old.
  Fact 0.1419: Walker, Mr. William Anderson is from East Orange, New Jersey, US.
  Fact 0.1420: Walker, Mr. William Anderson boarded at Southampton.
  Fact 0.1421: Walker, Mr. William Anderson's destination is East Orange, New Jersey, US.
  Fact 0.1422: Warren, Mr. Frank Manley was 63 years old.
  Fact 0.1423: Warren, Mr. Frank Manley is from Portland, Oregon, US.
  Fact 0.1424: Warren, Mr. Frank Manley boarded at Cherbourg.
  Fact 0.1425: Warren, Mr. Frank Manley's destination is Portland, Oregon, US.
  Fact 0.1426: Warren, Mrs. Anna Sophia (née Atkinson) was 60 years old.
  Fact 0.1427: Warren, Mrs. Anna Sophia (née Atkinson) is from Portland, Oregon, US.
  Fact 0.1428: Warren, Mrs. Anna Sophia (née Atkinson) boarded at Cherbourg.
  Fact 0.1429: Warren, Mrs. Anna Sophia (née Atkinson)'s destination is Portland, Oregon, US.
  Fact 0.1430: Warren, Mrs. Anna Sophia (née Atkinson) has 5 in the lifeboat or body column.
  Fact 0.1431: Weir, Colonel John was 59 years old.
  Fact 0.1432: Weir, Colonel John is from New York City.
  Fact 0.1433: Weir, Colonel John boarded at Southampton.
  Fact 0.1434: Weir, Colonel John's destination is New York City.
  Fact 0.1435: White, Mr. Percival Wayland was 54 years old.
  Fact 0.1436: White, Mr. Percival Wayland is from Brunswick, Maine, US.
  Fact 0.1437: White, Mr. Percival Wayland boarded at Southampton.
  Fact 0.1438: White, Mr. Percival Wayland's destination is Brunswick, Maine, US.
  Fact 0.1439: White, Mr. Richard Frasar was 21 years old.
  Fact 0.1440: White, Mr. Richard Frasar is from Brunswick, Maine, US.
  Fact 0.1441: White, Mr. Richard Frasar boarded at Southampton.
  Fact 0.1442: White, Mr. Richard Frasar's destination is Brunswick, Maine, US.
  Fact 0.1443: White, Mr. Richard Frasar has 169 in the lifeboat or body column.
  Fact 0.1444: White, Mrs. Ella (née Holmes) was 55 years old.
  Fact 0.1445: White, Mrs. Ella (née Holmes) is from New York City.
  Fact 0.1446: White, Mrs. Ella (née Holmes) boarded at Cherbourg.
  Fact 0.1447: White, Mrs. Ella (née Holmes)'s destination is New York City.
  Fact 0.1448: White, Mrs. Ella (née Holmes) has 8 in the lifeboat or body column.
  Fact 0.1449: Miss Amelia Mayo "Nellie" Bessette was 39 years old.
  Fact 0.1450: Miss Amelia Mayo "Nellie" Bessette is listed as the maid.
  Fact 0.1451: Miss Amelia Mayo "Nellie" Bessette is from New York City.
  Fact 0.1452: Miss Amelia Mayo "Nellie" Bessette boarded at Cherbourg.
  Fact 0.1453: Miss Amelia Mayo "Nellie" Bessette's destination is New York City.
  Fact 0.1454: Miss Amelia Mayo "Nellie" Bessette has 8 in the lifeboat or body column.
  Fact 0.1455: Mr. Sante Righini was 22 years old.
  Fact 0.1456: Mr. Sante Righini is listed as the manservant.
  Fact 0.1457: Mr. Sante Righini is from New York City.
  Fact 0.1458: Mr. Sante Righini boarded at Cherbourg.
  Fact 0.1459: Mr. Sante Righini's destination is New York City.
  Fact 0.1460: Mr. Sante Righini has 232 in the lifeboat or body column.
  Fact 0.1461: Wick, Colonel George Dennick was 58 years old.
  Fact 0.1462: Wick, Colonel George Dennick is from Youngstown, Ohio, US.
  Fact 0.1463: Wick, Colonel George Dennick boarded at Southampton.
  Fact 0.1464: Wick, Colonel George Dennick's destination is Youngstown, Ohio, US.
  Fact 0.1465: Wick, Colonel George Dennick has 8 in the lifeboat or body column.
  Fact 0.1466: Wick, Mrs. Mary (née Hitchcock) was 45 years old.
  Fact 0.1467: Wick, Mrs. Mary (née Hitchcock) is from Youngstown, Ohio, US.
  Fact 0.1468: Wick, Mrs. Mary (née Hitchcock) boarded at Southampton.
  Fact 0.1469: Wick, Mrs. Mary (née Hitchcock)'s destination is Youngstown, Ohio, US.
  Fact 0.1470: Wick, Mrs. Mary (née Hitchcock) has 8 in the lifeboat or body column.
  Fact 0.1471: Wick, Miss Mary Natalie was 31 years old.
  Fact 0.1472: Wick, Miss Mary Natalie is from Youngstown, Ohio, US.
  Fact 0.1473: Wick, Miss Mary Natalie boarded at Southampton.
  Fact 0.1474: Wick, Miss Mary Natalie's destination is Youngstown, Ohio, US.
  Fact 0.1475: Wick, Miss Mary Natalie has 8 in the lifeboat or body column.
  Fact 0.1476: Widener, Mr. George Dunton was 50 years old.
  Fact 0.1477: Widener, Mr. George Dunton is from Philadelphia, Pennsylvania, US.
  Fact 0.1478: Widener, Mr. George Dunton boarded at Southampton.
  Fact 0.1479: Widener, Mr. George Dunton's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1480: Mr. Edwin Herbert Keeping was 33 years old.
  Fact 0.1481: Mr. Edwin Herbert Keeping is listed as the valet.
  Fact 0.1482: Mr. Edwin Herbert Keeping is from Philadelphia, Pennsylvania, US.
  Fact 0.1483: Mr. Edwin Herbert Keeping boarded at Southampton.
  Fact 0.1484: Mr. Edwin Herbert Keeping's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1485: Mr. Edwin Herbert Keeping has 45 in the lifeboat or body column.
  Fact 0.1486: Widener, Mrs. Eleanor (née Elkins) was 50 years old.
  Fact 0.1487: Widener, Mrs. Eleanor (née Elkins) is from Philadelphia, Pennsylvania, US.
  Fact 0.1488: Widener, Mrs. Eleanor (née Elkins) boarded at Southampton.
  Fact 0.1489: Widener, Mrs. Eleanor (née Elkins)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1490: Miss Amalie Henriette "Emily" Gieger was 35 years old.
  Fact 0.1491: Miss Amalie Henriette "Emily" Gieger is listed as the maid.
  Fact 0.1492: Miss Amalie Henriette "Emily" Gieger is from Philadelphia, Pennsylvania, US.
  Fact 0.1493: Miss Amalie Henriette "Emily" Gieger boarded at Southampton.
  Fact 0.1494: Miss Amalie Henriette "Emily" Gieger's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1495: Widener, Mr. Harry Elkins was 27 years old.
  Fact 0.1496: Widener, Mr. Harry Elkins is from Philadelphia, Pennsylvania, US.
  Fact 0.1497: Widener, Mr. Harry Elkins boarded at Southampton.
  Fact 0.1498: Widener, Mr. Harry Elkins's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1499: Willard, Miss Constance was 21 years old.
  Fact 0.1500: Willard, Miss Constance is from Duluth, Minnesota, US.
  Fact 0.1501: Willard, Miss Constance boarded at Southampton.
  Fact 0.1502: Willard, Miss Constance's destination is Duluth, Minnesota, US.
  Fact 0.1503: Willard, Miss Constance has 8 in the lifeboat or body column.
  Fact 0.1504: Williams, Mr. Charles Duane was 51 years old.
  Fact 0.1505: Williams, Mr. Charles Duane is from Geneva, Switzerland.
  Fact 0.1506: Williams, Mr. Charles Duane boarded at Cherbourg.
  Fact 0.1507: Williams, Mr. Charles Duane's destination is Radnor, Pennsylvania, US.
  Fact 0.1508: Williams, Mr. Richard Norris II was 21 years old.
  Fact 0.1509: Williams, Mr. Richard Norris II is from Geneva, Switzerland.
  Fact 0.1510: Williams, Mr. Richard Norris II boarded at Cherbourg.
  Fact 0.1511: Williams, Mr. Richard Norris II's destination is Radnor, Pennsylvania, US.
  Fact 0.1512: Woolner, Mr. Hugh was 45 years old.
  Fact 0.1513: Woolner, Mr. Hugh is from London, England, UK.
  Fact 0.1514: Woolner, Mr. Hugh boarded at Southampton.
  Fact 0.1515: Woolner, Mr. Hugh's destination is New York City.
  Fact 0.1516: Wright, Mr. George was 62 years old.
  Fact 0.1517: Wright, Mr. George is from Halifax, Nova Scotia, Canada.
  Fact 0.1518: Wright, Mr. George boarded at Southampton.
  Fact 0.1519: Wright, Mr. George's destination is Halifax, Nova Scotia, Canada.
  Fact 0.1520: Young, Miss Marie Grice was 36 years old.
  Fact 0.1521: Young, Miss Marie Grice is from New York City.
  Fact 0.1522: Young, Miss Marie Grice boarded at Cherbourg.
  Fact 0.1523: Young, Miss Marie Grice's destination is New York City.
  Fact 0.1524: Young, Miss Marie Grice has 8 in the lifeboat or body column.

Retrieved items:
<item 1>
Passengers of the RMS Titanic
Frank M. Warren , Sr . George Dennick Wick George Dunton Widener Harry Elkins Widener Duane Williams George Henry Wright Roderick Chisholm Survivors ( last
living ) Rhoda Abbott Trevor Allison Lillian Asplund Madeleine Astor Ruth Becker Lawrence Beesley Karl Behr Dickinson Bishop Mauritz Håkan Björnström -
Steffansson Elsie Bowerman Francis Browne Margaret `` Molly '' Brown Daniel Buckley Alden Caldwell Helen Churchill Candee Charlotte Drake Cardeza Lucile Carter
Gladys Cherry Millvina Dean Margaret Devaney Sir Cosmo Duff - Gordon Lucy , Lady Duff - Gordon Ethel Flora Fortune Dorothy Gibson Archibald Gracie IV Frank John
William Goldsmith Edith Haisman Henry S. Harper Eva Hart Margaret Bechstein Hays Masabumi Hosono J. Bruce Ismay Eleanor Ileen Johnson Louise Laroche Louise Kink
Margaret Mannion Michel Marcel Navratil Alfred Nourney Arthur Godfrey Peuchen Winnifred Quick Marjorie Newell Robb Edith Rosenbaum Noël Leslie ,
</item 1>

<item 2>
Passengers of the RMS Titanic
Of the 711 passengers and crew rescued by the Carpathia , six , including first class passenger William F. Hoyt , either died in a lifeboat during the night or
on board the Carpathia the next morning , and were buried at sea . In the days following the sinking , several ships sailed to the disaster area to recover
victims ' bodies . The White Star Line chartered the cable ship Mackay - Bennett from Halifax , Nova Scotia to retrieve bodies . Three other ships followed in
the search : the cable ship Minia , the lighthouse supply ship Montmagny and the sealing vessel Algerine . Each ship left with embalming supplies , undertakers
, and clergy . Upon recovery , each body retrieved by the Mackay - Bennett was numbered and given as detailed a description as possible to help aid in
identification . The physical appearance of each body -- height , weight , age , hair and eye colour , visible birthmarks , scars or tattoos , was catalogued
and any personal effects on the bodies were gathered and placed in small canvas bags corresponding to their number .
</item 2>

<item 3>
Passengers of the RMS Titanic
1 British inquiry US inquiry Wreck of Titanic Maritime Memorial Act Deck officers Edward J. Smith ( Captain ) Henry Tingle Wilde ( Chief Officer ) William
McMaster Murdoch ( First Officer ) Charles H. Lightoller ( Second Officer ) Herbert Pitman ( Third Officer ) Joseph G. Boxhall ( Fourth Officer ) Harold G. Lowe
( Fifth Officer ) James Paul Moody ( Sixth Officer ) Joseph Bell ( Machine Room Manager ) Crew members Frederick Barrett Harold Bride William Denton Cox Sid
Daniels Alfred Frank Evans Frederick Fleet Luigi Gatti Robert Hichens Violet Jessop Charles Joughin Reginald Lee Evelyn Marsden William Mintram Jack Phillips
George Symons Passengers Fatalities Allison family Thomas Andrews John Jacob Astor IV David John Bowen Archibald Butt Thomas Byles Walter Donald Douglas Edith
Corse Evans Annie Funk Jacques Futrelle Sidney Leslie Goodwin Benjamin Guggenheim John Harper Wallace Hartley Charles Melville Hays Edward Austin Kent Joseph
Philippe Lemercier Laroche Francis Davis Millet Harry Markland Molson Michel Navratil Eino Viljami Panula W.T. Stead Ida Straus Isidor Straus John Borland
Thayer Jr . Frank M. Warren , Sr .
</item 3>

<item 4>
Passengers of the RMS Titanic
ISBN 978 - 0 - 14 - 004757 - 8 . Jump up ^ `` Passenger List and Survivors of Steamship Titanic '' . United States Senate Inquiry. 30 July 1912 . Retrieved 15
July 2011 . Jump up ^ Hall , Wayne ( 1986 ) . `` Social Class and Survival on the SS Titanic '' ( PDF ) . Social Science & Medicine . 22 ( 6 ) : 687 -- 690 .
Jump up ^ Barratt , Nick ( 2009 ) . Lost Voices From the Titanic : The Definitive Oral History . London : Random House . p. 93 . ISBN 978 - 1 - 84809 - 151 - 1
. Jump up ^ Howells , Richard ( 1999 ) . The Myth of the Titanic . United Kingdom : MacMillan Press . p. 18 . ISBN 978 - 0 - 333 - 72597 - 9 .
</item 4>

<item 5>
Passengers of the RMS Titanic
Victims whose remains were recovered after the sinking are listed with a superscript next to the body number , indicating the recovery vessel : MB -- CS Mackay
- Bennett ( bodies 1 -- 306 ) M -- CS Minia ( bodies 307 -- 323 ) MM -- CGS Montmagny ( bodies 326 -- 329 ) A -- SS Algerine ( body 330 ) O -- RMS Oceanic (
bodies 331 -- 333 ) I -- SS Ilford ( body 334 ) OT -- SS Ottawa ( body 335 ) Numbers 324 and 325 were unused , and the six bodies buried at sea by the Carpathia
also went unnumbered . First class ( edit ) Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US
Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and
chauffeur , Mr.
</item 5>

<item 6>
Passengers of the RMS Titanic
Although several people managed to reach this lifeboat , three died during the night . When Fifth Officer Harold Lowe and six crewmen returned to the wreck site
after the sinking with an empty lifeboat to pick up survivors , they rescued surviving passengers from Collapsible A , but left the three dead bodies in the
boat : Thomson Beattie , a first - class passenger , and two crew members , a fireman and a seaman . After their retrieval from Collapsible A by Oceanic , the
bodies were buried at sea . Passenger list ( edit ) Breakdown of casualties according to the British Board of Trade report ( click for detail ) The following is
a full list of known passengers who sailed on the maiden voyage of the RMS Titanic . Included in this list are the nine - member Guarantee Group and the eight
members of the ship 's band , listed as both passengers and crew . They are also included in the list of crew members on board RMS Titanic . Passengers are
colour - coded , indicating whether they were saved or perished . The passenger did not survive The passenger survived Survivors are listed with the lifeboat
from which they were known to be rescued .
</item 6>

<item 7>
Passengers of the RMS Titanic
( Halifax ) Related Ships RMS Olympic RMS Carpathia HMHS Britannic CS Mackay - Bennett SS Californian Replica Titanic Titanic II Others White Star Line Titanic
Historical Society Encyclopedia Titanica Halomonas titanicae Women and children first SOS Robert Ballard Last remaining survivors of the RMS Titanic disaster
2000 -- 2009 Millvina Dean ( 1912 -- 2009 ) Barbara West ( 1911 -- 2007 ) Lillian Asplund ( 1906 -- 2006 ) Winnifred Quick ( 1904 -- 2002 ) Michel Navratil (
1908 -- 2001 ) 1995 -- 1999 Eleanor Johnson ( 1910 -- 1998 ) Louise Laroche ( 1910 -- 1998 ) Edith Brown ( 1896 -- 1997 ) Eva Hart ( 1905 -- 1996 ) Beatrice
Sandström ( 1910 -- 1995 ) 1990 -- 1994 Robertha `` Bertha '' Watt ( 1899 -- 1993 ) Ellen Shine ( 1891 -- 1993 ) Marjorie Newell ( 1889 -- 1992 ) Louise
</item 7>

<item 8>
Passengers of the RMS Titanic
Jump up ^ Brewster , Hugh ; Coulter , Laurie ( 1998 ) . 882 1 / 2 Amazing Answers To Your Questions About The Titanic . Scholastic Paperbacks . p. 18 . ISBN 978
- 0 - 8117 - 1814 - 1 . Jump up ^ I Survived the Titanic : Ruth Becker Jump up ^ `` Whitaker Centre -- ' ' Titanic Artifact Exhibit : Ministry Through Tragedy '
' '' ( PDF ) . Archived from the original ( PDF ) on 4 July 2011 . Retrieved 6 July 2011 . Jump up ^ `` Fr Thomas Roussel Davids Byles '' . Encyclopedia
Titanica . Retrieved 15 July 2011 . Jump up ^ `` Pastor John Harper and the Titanic Story '' . Harper Memorial Baptist Church . Archived from the original on 17
April 2012 . Retrieved 2 April 2012 . Jump up ^ `` Lawrence Beesley -- Celebrated Titanic Survivor '' . Bygonederbyshire.co.uk . Retrieved 6 July 2011 .
</item 8>

<item 9>
Passengers of the RMS Titanic
Jump up ^ William Hoyt was pulled alive from the water , and onto Lifeboat 14 , and thus is counted among the occupants of this boat ; Hoyt , however , died
during the night and his body was buried at sea by the sailors on board the Carpathia Jump up ^ `` One Victim Legally Dead '' . The New York Times . 27 April
1912 . p. 4 < ! None > Jump up ^ Marion Kenyon was pregnant while on board the Titanic . She delivered a stillborn child sometime in 1912 . Jump up ^ `` Miss
Roberta Elizabeth Mary Maioni '' . Encyclopedia-titanica.org . Retrieved 6 July 2011 . Jump up ^ Mary Marvin was pregnant while on board the Titanic . She gave
birth to a daughter , Margaret , sometime in 1912 . Jump up ^ Traveled using alias : Baron Alfred von Drachstedt ; transferred from second to first class Jump
up ^ ( 2013 ) Alfred Nourney ( ref : # 220 ) Encyclopedia Titanica , ( accessed 17 December 2013 ) Jump up ^ Eloise Smith was pregnant while on board the
Titanic .
</item 9>

<item 10>
Passengers of the RMS Titanic
Collapsible B floated away from the ship upside down , while Collapsible A became half - filled with water after the supports for its canvas sides were broken
in the fall from the roof of the officers ' quarters . There were arguments in some of the lifeboats about going back to pick up people in the water , but many
survivors were afraid of being swamped by people trying to climb into the lifeboat or being pulled down by the suction from the sinking Titanic , though it
turned out that there had been very little suction . At 2 : 20 am , Titanic herself sank . A small number of passengers and crew were able to make their way to
the two unlaunched collapsible boats , surviving for several hours ( some still clinging to the overturned Collapsible B) until they were rescued by Fifth
Officer Harold Lowe . The RMS Carpathia arriving at Pier 54 with Titanic survivors . At 4 : 10 am , the RMS Carpathia arrived at the site of the sinking and
began rescuing survivors . By 8 : 30 am , she picked up the last lifeboat with survivors and left the area at 08 : 50 bound for Pier 54 in New York City .
</item 10>
=== END ITEM id=d195eadda113b55c ===

=== ITEM id=cf0855b09ff6ba3e ===
Gold sentences and facts:
Sentence 0: Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
  Fact 0.0: Uncle Tom's Cabin was serialized in The National Era for ten months.
  Fact 0.1: The serialization of Uncle Tom's Cabin in The National Era began with the June 5, 1851 issue.
  Fact 0.2: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852.
  Fact 0.3: The book edition of Uncle Tom's Cabin was published by John P. Jewett and Company.
Sentence 1: Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical , starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story 's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's Cabin in book form , she eventually consented to the request .
  Fact 1.0: Uncle Tom's Cabin first appeared as a 40-week serial in The National Era.
  Fact 1.1: The National Era was an abolitionist periodical.
  Fact 1.2: The serial started with the June 5, 1851 issue.
  Fact 1.3: Uncle Tom's Cabin was originally intended as a shorter narrative that would run for only a few weeks.
  Fact 1.4: Harriet Beecher Stowe expanded the story significantly.
  Fact 1.5: The expanded story was instantly popular.
  Fact 1.6: Several protests were sent to the National Era office when Stowe missed an issue.
  Fact 1.7: The publisher John P. Jewett contacted Stowe about turning the serial into a book because of the story's popularity.
  Fact 1.8: Stowe questioned whether anyone would read Uncle Tom's Cabin in book form.
  Fact 1.9: Stowe eventually consented to the request to publish the story as a book.
Sentence 2: Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or , Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company . Followed by A Key to Uncle Tom 's Cabin ( 1853 )
  Fact 2.0: The infobox author of Uncle Tom's Cabin is Harriet Beecher Stowe.
  Fact 2.1: The original title of Uncle Tom's Cabin is Uncle Tom's Cabin; or, Life Among the Lowly.
  Fact 2.2: The illustrator of the 1st edition of Uncle Tom's Cabin is Hammatt Billings.
  Fact 2.3: Uncle Tom's Cabin is from the United States.
  Fact 2.4: Uncle Tom's Cabin is written in English.
  Fact 2.5: Uncle Tom's Cabin is a novel.
  Fact 2.6: Uncle Tom's Cabin was serialized in The National Era for ten months, beginning with the June 5, 1851 issue.
  Fact 2.7: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852, by John P. Jewett and Company.
  Fact 2.8: Uncle Tom's Cabin is followed by A Key to Uncle Tom's Cabin (1853).
  Fact 2.9: The infobox includes an image of the Boston edition of Uncle Tom's Cabin.

Retrieved items:
<item 1>
Uncle Tom's Cabin
Later , it was translated into almost every major language , including Chinese ( with translator Lin Shu creating the first Chinese translation of an American
novel in 1901 ) and Amharic ( with the 1930 translation created in support of Ethiopian efforts to end the suffering of blacks in that nation ) . The book was
so widely read that Sigmund Freud reported a number of patients with sado - masochistic tendencies who he believed had been influenced by reading about the
whipping of slaves in Uncle Tom 's Cabin . A short article which was published in the oficial newspaper of the Salvadoran government in July 22 , 1853 praised
Harriet Beecher Stowe and her book 's success . Literary significance and criticism As the first widely read political novel in the United States , Uncle Tom 's
Cabin greatly influenced development of not only American literature but also protest literature in general . Later books which owe a large debt to Uncle Tom 's
Cabin include The Jungle by Upton Sinclair and Silent Spring by Rachel Carson . Eliza crosses the Ohio on the cover of Pictures and Stories from Uncle Tom 's
Cabin , Boston : John P. Jewett & Co. , 1853 .
</item 1>

<item 2>
Uncle Tom's Cabin
In the decade between the publication of Uncle Tom 's Cabin and the start of the American Civil War , between twenty and thirty anti-Tom books were published .
Among these novels are two books titled Uncle Tom 's Cabin As It Is ( one by W.L. Smith and the other by C.H. Wiley ) and a book by John Pendleton Kennedy .
More than half of these anti-Tom books were written by white women , with Simms commenting at one point about the `` Seemingly poetic justice of having the
Northern woman ( Stowe ) answered by a Southern woman . '' Dramatic adaptations George Aiken 's original manuscript for his stage adaptation of Uncle Tom 's
Cabin , 1852 . From the George C. Howard and Family Collection at the Harry Ransom Center . Tom shows Main article : Tom show Even though Uncle Tom 's Cabin was
the best - selling novel of the 19th century , far more Americans of that time saw the story as a stage play or musical than read the book .
</item 2>

<item 3>
Uncle Tom's Cabin
4 ( January 1960 ) , pp. 421 -- 433 . Jump up ^ The Classic Text : Harriett Beecher Stowe . University of Wisconsin - Milwaukee Library . Special collection
page on traditions and interpretations of Uncle Tom 's Cabin . Retrieved May 15 , 2007 . Archived May 16 , 2008 , at the Wayback Machine . ^ Jump up to : Uncle
Tom 's Cabin , introduction by Amanda Claybaugh , Barnes and Noble Classics , New York , 2003 , p. xvii . Jump up ^ A.M. Arnett , `` Review of James Baird
Weaver by Fred Emory Haynes '' , Political Science Quarterly , Vol. 35 , No. 1 ( March 1920 ) , pp. 154 -- 157 ; and profile of James Baird Weaver Archived
January 2 , 2007 , at the Wayback Machine. , accessed February 17 , 2007 . Jump up ^ Nassau Senior , quoted in Ephraim Douglass Adams , Great Britain and the
American Civil War ( 1958 ) , p. 33 .
</item 3>

<item 4>
Uncle Tom's Cabin
13 . ^ Jump up to : Darryl Lorenzo Wellington , `` Uncle Tom 's Shadow '' , The Nation , December 25 , 2006 . Jump up ^ Smylie , James H. `` Uncle Tom 's Cabin
Revisited : the Bible , the Romantic Imagination , and the Sympathies of Christ . '' American Presbyterians 1995 73 ( 3 ) : 165 -- 175 . ISSN 0886 - 5159 . ^
Jump up to : Bellin , Joshua D. `` Up to Heaven 's Gate , down in Earth 's Dust : the Politics of Judgment in Uncle Tom 's Cabin '' , American Literature 1993
65 ( 2 ) : 275 -- 295 . ISSN 0002 - 9831 Fulltext online at Jstor and Ebsco . ^ Jump up to : Grant , David , `` Uncle Tom 's Cabin and the Triumph of Republican
Rhetoric '' , New England Quarterly 1998 71 ( 3 ) : 429 -- 448 . ISSN 0028 - 4866 Fulltext online at Jstor .
</item 4>

<item 5>
Uncle Tom's Cabin
Uncle Tom 's Cabin : or Life among the lowly ; frontispiece by John Gilbert ; ornamental title - page by Phiz ; and 130 engravings on wood by Matthew Urlwin
Sears , 1853 ( a searchable facsimile at the University of Georgia Libraries ; DjVu & layered PDF format ) Pictures and stories from Uncle Tom 's Cabin ; `` The
purpose of the editor of this little work , has been to adapt it for the juvenile family circle . The verses have accordingly been written by the authoress for
the capacity of the youngest readers ... '' 1853 ( a searchable facsimile at the University of Georgia Libraries ; DjVu & layered PDF format ) Uncle Tom 's
Cabin at Project Gutenberg Uncle Tom 's Cabin , available at Internet Archive . Scanned , illustrated original editions .
</item 5>

<item 6>
Uncle Tom's Cabin
Uncle Tom 's Cabin - wikipedia Uncle Tom 's Cabin Jump to : navigation , search This article is about the mid-19th century novel . For other uses , see Uncle
Tom 's Cabin ( disambiguation ) . Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or ,
Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era
for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
</item 6>

<item 7>
Uncle Tom's Cabin
The book was translated into all major languages , and in the United States it became the second best - selling book after the Bible . A number of the early
editions carried an introduction by Rev James Sherman , a Congregational minister in London noted for his abolitionist views . Uncle Tom 's Cabin sold equally
well in Britain , with the first London edition appearing in May 1852 and selling 200,000 copies . In a few years over 1.5 million copies of the book were in
circulation in Britain , although most of these were infringing copies ( a similar situation occurred in the United States ) . Plot Eliza escapes with her son ,
Tom sold `` down the River '' Full - page illustration by Hammatt Billings for Uncle Tom 's Cabin depicts Eliza telling Uncle Tom that he has been sold and she
is running away to save her child ( first edition : Boston : John P. Jewett and Company , 1852 ) . The book opens with a Kentucky farmer named Arthur Shelby
facing the loss of his farm because of debts .
</item 7>

<item 8>
Uncle Tom's Cabin
Stowe mentioned a number of the inspirations and sources for her novel in A Key to Uncle Tom 's Cabin ( 1853 ) . This non-fiction book was intended to verify
Stowe 's claims about slavery . However , later research indicated that Stowe did not read many of the book 's cited works until after she had published her
novel . Publication The National Era , June 5 , 1851 Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical ,
starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story
significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story
's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's
Cabin in book form , she eventually consented to the request .
</item 8>

<item 9>
Uncle Tom's Cabin
Convinced the book would be popular , Jewett made the unusual decision ( for the time ) to have six full - page illustrations by Hammatt Billings engraved for
the first printing . Published in book form on March 20 , 1852 , the novel sold 3,000 copies on that day alone , and soon sold out its complete print run . A
number of other editions were soon printed ( including a deluxe edition in 1853 , featuring 117 illustrations by Billings ) . In the first year of publication ,
300,000 copies of Uncle Tom 's Cabin were sold . At that point , however , `` demand came to an unexpected halt ... No more copies were produced for many years
, and if , as is claimed , Abraham Lincoln greeted Stowe in 1862 as ' the little woman who wrote the book that made this great war , ' the work had effectively
been out of print for many years . '' Jewett went out of business , and it was not until Ticknor and Fields put the work back in print in November 1862 that
demand began again to increase .
</item 9>

<item 10>
Uncle Tom's Cabin
42 . ^ Jump up to : Thomas F. Gossett , `` Review of The Building of Uncle Tom 's Cabin by E. Bruce Kirkham '' , American Literature , Vol. 50 , No. 1 ( March
1978 ) , pp. 123 -- 124 . Jump up ^ Charles Nichols , `` The Origins of Uncle Tom 's Cabin '' , The Phylon Quarterly , Vol. 19 , No. 3 ( 3rd Qtr , 1958 ) , p.
328 . ^ Jump up to : Jane Tompkins , `` Sentimental Power : Uncle Tom 's Cabin and the Politics of Literary History '' Archived December 16 , 2007 , at the
Wayback Machine. , from In Sensational Designs : The Cultural Work of American Fiction , 1790 -- 1860 . New York : Oxford University Press , 1985 , pp. 122 --
146 . Jump up ^ Posner , R. , Public Intellectuals : A Study of Decline , Harvard University Press ( 2002 ) , p. 239 . ISBN 0 - 674 - 00633 - X .
</item 10>
=== END ITEM id=cf0855b09ff6ba3e ===

=== ITEM id=e753040195245d08 ===
Gold sentences and facts:
Sentence 0: The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional town of West Egg on prosperous Long Island in the summer of 1922 . The story primarily concerns the young and mysterious millionaire Jay Gatsby and his quixotic passion and obsession for the beautiful former debutante Daisy Buchanan . Considered to be Fitzgerald 's magnum opus , The Great Gatsby explores themes of decadence , idealism , resistance to change , social upheaval , and excess , creating a portrait of the Jazz Age or the Roaring Twenties that has been described as a cautionary tale regarding the American Dream .
  Fact 0.0: The Great Gatsby is a 1925 novel.
  Fact 0.1: The Great Gatsby was written by American author F. Scott Fitzgerald.
  Fact 0.2: The Great Gatsby follows a cast of characters living in the fictional town of West Egg on prosperous Long Island in the summer of 1922.
  Fact 0.3: The fictional town of West Egg is on prosperous Long Island.
  Fact 0.4: The story of The Great Gatsby primarily concerns the young and mysterious millionaire Jay Gatsby.
  Fact 0.5: The story of The Great Gatsby concerns Jay Gatsby's quixotic passion and obsession for the beautiful former debutante Daisy Buchanan.
  Fact 0.6: Daisy Buchanan is a beautiful former debutante.
  Fact 0.7: The Great Gatsby is considered to be Fitzgerald's magnum opus.
  Fact 0.8: The Great Gatsby explores themes of decadence, idealism, resistance to change, social upheaval, and excess.
  Fact 0.9: The Great Gatsby creates a portrait of the Jazz Age, also called the Roaring Twenties.
  Fact 0.10: The Great Gatsby has been described as a cautionary tale regarding the American Dream.
Sentence 1: Set on the prosperous Long Island of 1922 , The Great Gatsby provides a critical social history of America during the Roaring Twenties within its fictional narrative . That era , known for unprecedented economic prosperity , the evolution of jazz music flapper culture , new technologies in communication ( motion pictures , broadcast radio , recorded music ) forging a genuine mass culture ; and bootlegging , along with other criminal activity , is plausibly depicted in Fitzgerald 's novel . Fitzgerald uses many of these societal developments of the 1920s that were to build Gatsby 's stories from many of the simple details like automobiles to broader themes like Fitzgerald 's discreet allusions to the organized crime culture which was the source of Gatsby 's fortune . Fitzgerald depicts the garish society of the Roaring Twenties by placing the book 's plotline within the historical context of the era .
  Fact 1.0: The Great Gatsby is set on the prosperous Long Island of 1922.
  Fact 1.1: The Great Gatsby provides a critical social history of America during the Roaring Twenties within its fictional narrative.
  Fact 1.2: The Roaring Twenties era was known for unprecedented economic prosperity.
  Fact 1.3: The Roaring Twenties era was known for the evolution of jazz music and flapper culture.
  Fact 1.4: The Roaring Twenties era saw new technologies in communication: motion pictures, broadcast radio, and recorded music.
  Fact 1.5: The new communication technologies of the Roaring Twenties forged a genuine mass culture.
  Fact 1.6: The Roaring Twenties era saw bootlegging, along with other criminal activity.
  Fact 1.7: Bootlegging and other criminal activity of the Roaring Twenties are plausibly depicted in Fitzgerald's novel.
  Fact 1.8: Fitzgerald uses many societal developments of the 1920s to build the stories of Gatsby.
  Fact 1.9: Fitzgerald uses these societal developments in his novel, from simple details like automobiles to broader themes.
  Fact 1.10: Fitzgerald makes discreet allusions in his novel to the organized crime culture.
  Fact 1.11: The organized crime culture was the source of Gatsby's fortune.
  Fact 1.12: Fitzgerald depicts the garish society of the Roaring Twenties.
  Fact 1.13: Fitzgerald depicts the garish society by placing the book's plotline within the historical context of the era.

Retrieved items:
<item 1>
The Great Gatsby
Fitzgerald uses many of these societal developments of the 1920s that were to build Gatsby 's stories from many of the simple details like automobiles to
broader themes like Fitzgerald 's discreet allusions to the organized crime culture which was the source of Gatsby 's fortune . Fitzgerald depicts the garish
society of the Roaring Twenties by placing the book 's plotline within the historical context of the era . Fitzgerald 's visits to Long Island 's north shore
and his experience attending parties at mansions inspired The Great Gatsby 's setting . Today , there are a number of theories as to which mansion was the
inspiration for the book . One possibility is Land 's End , a notable Gold Coast Mansion where Fitzgerald may have attended a party . Many of the events in
Fitzgerald 's early life are reflected throughout The Great Gatsby . Fitzgerald was a young man from Minnesota , and like Nick , he was educated at an Ivy
League school , Princeton ( in Nick 's case , Yale ) .
</item 1>

<item 2>
The Great Gatsby
They introduce Nick to Jordan Baker , an attractive , cynical young golfer with whom Nick begins a romantic relationship . She reveals to Nick that Tom has a
mistress , Myrtle Wilson , who lives in the `` valley of ashes '' , an industrial dumping ground between West Egg and New York City . Not long after this
revelation , Nick travels to New York City with Tom and Myrtle to an apartment Tom keeps for his affairs with Myrtle and others . At Tom 's New York apartment ,
a vulgar and bizarre party takes place . It ends with Tom breaking Myrtle 's nose after she annoys him by saying Daisy 's name several times . The Plaza Hotel
in the early - 1920s As the summer progresses , Nick eventually receives an invitation to one of Gatsby 's parties . Nick encounters Jordan Baker at the party ,
and they meet Gatsby himself , an aloof and surprisingly young man who recognizes Nick from their same division in the Great War . Through Jordan , Nick later
learns that Gatsby knew Daisy through a purely chance meeting in 1917 , when Daisy and her friends were doing volunteer services ' work with young Officers
headed to Europe .
</item 2>

<item 3>
The Great Gatsby
The New York Times . Jump up ^ Bruccoli 2000 , pp. 38 -- 39 Jump up ^ Bruccoli 2000 , p. 45 Jump up ^ Randall , Mónica ( 2003 ) . The Mansions of Long Island 's
Gold Coast . Rizzoli . pp. 275 -- 277 . ISBN 978 - 0 - 8478 - 2649 - 0 . Jump up ^ West , James L.W. , III ( 2000 ) . Trimalchio : An Early Version of The Great
Gatsby . The Cambridge Edition of the Works of F. Scott Fitzgerald . Cover Design by Dennis M. Arnold . Cambridge University Press . p. xi . ISBN 0 - 521 -
89047 - 0 . Retrieved 27 July 2013 . He produces 18,000 words ; most of this material is later discarded , but he salvages the short story `` Absolution , ''
published in June 1924 . Jump up ^ Haglund , David ( 7 May 2013 ) . `` The Forgotten Childhood of Jay Gatsby '' .
</item 3>

<item 4>
The Great Gatsby
Considered to be Fitzgerald 's magnum opus , The Great Gatsby explores themes of decadence , idealism , resistance to change , social upheaval , and excess ,
creating a portrait of the Jazz Age or the Roaring Twenties that has been described as a cautionary tale regarding the American Dream . Fitzgerald -- inspired
by the parties he had attended while visiting Long Island 's north shore -- began planning the novel in 1923 , desiring to produce , in his words , `` something
new -- something extraordinary and beautiful and simple and intricately patterned . '' Progress was slow , with Fitzgerald completing his first draft following
a move to the French Riviera in 1924 . His editor , Maxwell Perkins , felt the book was vague and persuaded the author to revise over the following winter .
Fitzgerald was repeatedly ambivalent about the book 's title and he considered a variety of alternatives , including titles that referenced the Roman character
Trimalchio ; the title he was last documented to have desired was Under the Red , White , and Blue .
</item 4>

<item 5>
The Great Gatsby
In this novel , Great Neck ( King 's Point ) became the `` new money '' peninsula of West Egg and Port Washington ( Sands Point ) the old - money East Egg .
Several mansions in the area served as inspiration for Gatsby 's home , such as Oheka Castle and Beacon Towers , since demolished . By mid-1923 , Fitzgerald had
written 18,000 words for his novel but discarded most of his new story as a false start , some of which resurfaced in the 1924 short story `` Absolution '' .
Work on The Great Gatsby began in earnest in April 1924 . Fitzgerald wrote in his ledger , `` Out of woods at last and starting novel . '' He decided to make a
departure from the writing process of his previous novels and told Perkins that the novel was to be a `` consciously artistic achievement '' and a `` purely
creative work -- not trashy imaginings as in my stories but the sustained imagination of a sincere and yet radiant world . '' He added later , during editing ,
that he felt `` an enormous power in me now , more than I 've ever had . ''
</item 5>

<item 6>
The Great Gatsby
Nick stages an unsettlingly small funeral for Gatsby in which none of Gatsby 's associates or partygoers attend . Later , Nick runs into Tom in New York and
finds out that Tom had told George that Gatsby was Myrtle 's secret lover and that Gatsby had killed her , then gave Gatsby 's address to George . Nick breaks
up with Jordan , and , disillusioned with the East , moves back to the Midwest . Major characters ( edit ) Nick Carraway -- a Yale graduate originating from the
Midwest , a World War I veteran , and , at the start of the plot , a newly arrived resident of West Egg , who is aged 29 ( later 30 ) . He also serves as the
first - person narrator of the novel . He is Gatsby 's next - door neighbor and a bond salesman . He is easy - going , occasionally sarcastic , and somewhat
optimistic , although this latter quality fades as the novel progresses . He is a more grounded character than the others , and more practical , and is always
in awe of other character 's lifestyles and morals .
</item 6>

<item 7>
The Great Gatsby
Contents ( hide ) 1 Historical context 2 Plot summary 3 Major characters 4 Writing and production 5 Cover art 6 Title 7 Themes 8 Reception 9 Legacy and modern
analysis 10 Adaptations 10.1 Ballet 10.2 Computer games 10.3 Film and television 10.4 Literature 10.5 Opera 10.6 Radio 10.7 Theater 11 See also 12 Notes 13
References 13.1 Bibliography 14 External links Historical context ( edit ) Set on the prosperous Long Island of 1922 , The Great Gatsby provides a critical
social history of America during the Roaring Twenties within its fictional narrative . That era , known for unprecedented economic prosperity , the evolution of
jazz music flapper culture , new technologies in communication ( motion pictures , broadcast radio , recorded music ) forging a genuine mass culture ; and
bootlegging , along with other criminal activity , is plausibly depicted in Fitzgerald 's novel .
</item 7>

<item 8>
The Great Gatsby
The play failed miserably , and Fitzgerald worked that winter on magazine stories struggling to pay his debt caused by the production . The stories were , in
his words , `` all trash and it nearly broke my heart , '' although included among those stories was `` Winter Dreams '' , which Fitzgerald later described as
`` a sort of first draft of the Gatsby idea '' . After the birth of their child , the Fitzgeralds moved to Great Neck , New York , on Long Island , in October
1922 . The town was used as the scene of The Great Gatsby . Fitzgerald 's neighbors in Great Neck included such prominent and newly wealthy New Yorkers as
writer Ring Lardner , actor Lew Fields , and comedian Ed Wynn . These figures were all considered to be `` new money '' , unlike those who came from Manhasset
Neck or Cow Neck Peninsula , places which were home to many of New York 's wealthiest established families , and which sat across the bay from Great Neck . This
real - life juxtaposition gave Fitzgerald his idea for `` West Egg '' and `` East Egg '' .
</item 8>

<item 9>
The Great Gatsby
The Great Gatsby - Wikipedia The Great Gatsby Jump to : navigation , search This article is about the novel . For the film , TV and opera adaptations , see The
Great Gatsby ( disambiguation ) . The Great Gatsby Cover of the first edition in 1925 . Author F. Scott Fitzgerald Cover artist Francis Cugat Country United
States Language English Genre Novel Published April 10 , 1925 ( Charles Scribner 's Sons ) Media type Print ( hardcover & paperback ) Preceded by The Beautiful
and Damned ( 1922 ) Followed by Tender Is the Night ( 1934 ) The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast
of characters living in the fictional town of West Egg on prosperous Long Island in the summer of 1922 . The story primarily concerns the young and mysterious
millionaire Jay Gatsby and his quixotic passion and obsession for the beautiful former debutante Daisy Buchanan .
</item 9>

<item 10>
The Great Gatsby
In her book Careless People : Murder , Mayhem and the Invention of ' The Great Gatsby ( 2013 ) , Sarah Churchwell speculates that parts of the ending of The
Great Gatsby were based on the Hall - Mills Case . Based on her forensic search for clues , she asserts that the two victims in the Hall - Mills murder case
inspired the characters who were murdered in The Great Gatsby . Plot summary ( edit ) The main events of the novel take place in the summer of 1922 . Nick
Carraway , a Yale graduate and veteran of the Great War from the Midwest -- who serves as the novel 's narrator -- takes a job in New York as a bond salesman .
He rents a small house on Long Island , in the fictional village of West Egg , next door to the lavish mansion of Jay Gatsby , a mysterious multi-millionaire
who holds extravagant parties but does not participate in them . Nick drives around the bay to East Egg for dinner at the home of his cousin , Daisy Fay
Buchanan , and her husband , Tom , a college acquaintance of Nick 's .
</item 10>
=== END ITEM id=e753040195245d08 ===

=== ITEM id=ee5795dc3160be8f ===
Gold sentences and facts:
Sentence 0: ABO and Rh blood type distribution by country ( population averages ) Country Population O+ A+ B+ AB+ O − A − B − AB − Argentina 44,270,440 45.4 % 34.26 % 8.59 % 2.64 % 8.4 % 0.44 % 0.21 % 0.06 % Armenia 2,931,568 29.0 % 46.3 % 12.0 % 5.6 % 2.0 % 3.7 % 1.0 % 0.4 % Australia 24,642,693 40.0 % 31.0 % 8.0 % 2.0 % 9.0 % 7.0 % 2.0 % 1.0 % Austria 8,592,470 30.0 % 33.0 % 12.0 % 6.0 % 7.0 % 8.0 % 3.0 % 1.0 % Bahrain 1,418,695 48.48 % 19.35 % 22.61 % 3.67 % 3.27 % 1.33 % 1.04 % 0.25 % Bangladesh 164,833,667 31.18 % 21.44 % 34.58 % 8.85 % 1.39 % 0.96 % 0.96 % 0.64 % Belgium 11,444,053 38.0 % 34.0 % 8.6 % 4.1 % 7.0 % 6.0 % 1.5 % 0.8 % Bolivia 11,053,376 51.53 % 29.45 % 10.11 % 1.15 % 4.39 % 2.73 % 0.54 % 0.1 % Bosnia and Herzegovina 3,792,730 31.0 % 36.0 % 12.0 % 6.0 % 5.0 % 7.0 % 2.0 % 1.0 % Brazil 211,248,418 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Bulgaria 7,045,097 28.0 % 37.0 % 13.0 % 7.0 % 5.0 % 7.0 % 2.0 % 1.0 % Cambodia 16,077,172 46.7 % 27.2 % 18.5 % 4.9 % 1.3 % 0.8 % 0.5 % 0.1 % Cameroon 24,515,533 42.8 % 38.8 % 12.0 % 3.3 % 1.4 % 1.2 % 0.4 % 0.1 % Canada 36,627,140 39.0 % 36.0 % 7.6 % 2.5 % 7.0 % 6.0 % 1.4 % 0.5 % Chile 18,314,060 85.5 % 8.7 % 3.35 % 1.0 % 1.2 % 0.1 % 0.05 % 0.1 % China 1,388,251,023 47.7 % 27.8 % 18.9 % 5.0 % 0.28 % 0.19 % 0.1 % 0.03 % Colombia 49,069,267 61.3 % 26.11 % 2.28 % 1.47 % 5.13 % 2.7 % 0.7 % 0.31 % Côte d'Ivoire 23,869,656 46.5 % 22.5 % 22.5 % 4.3 % 2.0 % 1.0 % 1.0 % 0.2 % Croatia 4,207,355 29.0 % 36.0 % 15.0 % 5.0 % 5.0 % 6.0 % 3.0 % 1.0 % Cuba 11,486,750 45.8 % 33.5 % 10.2 % 2.9 % 3.6 % 2.8 % 1.0 % 0.2 % Cyprus 1,189,395 35.22 % 40.35 % 11.11 % 4.72 % 3.85 % 3.48 % 0.87 % 0.40 % Czech Republic 10,555,152 27.0 % 36.0 % 15.0 % 7.0 % 5.0 % 6.0 % 3.0 % 1.0 % Denmark 5,711,902 35.0 % 37.0 % 8.0 % 4.0 % 6.0 % 7.0 % 2.0 % 1.0 % Dominican Republic 10,766,932 46.2 % 26.4 % 16.9 % 3.1 % 3.7 % 2.1 % 1.4 % 0.2 % Ecuador 16,665,756 75.0 % 14.0 % 7.1 % 0.5 % 2.38 % 0.7 % 0.3 % 0.02 % Egypt 95,220,838 52.0 % 24.0 % 12.4 % 3.8 % 5.0 % 2.0 % 0.6 % 0.2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia 104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 % 37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Greece 10,892,849 37.4 % 32.9 % 11.0 % 3.7 % 7.0 % 5.0 % 2.0 % 1.0 % Guinea 13,291,741 46.88 % 21.64 % 22.86 % 4.52 % 2.0 % 0.9 % 1.0 % 0.2 % Honduras 9,348,898 57.5 % 27.0 % 7.8 % 2.5 % 2.7 % 1.7 % 0.6 % 0.2 % Hong Kong 7,402,115 41.5 % 26.13 % 25.34 % 6.35 % 0.32 % 0.17 % 0.14 % 0.05 % Hungary 9,787,801 27.0 % 33.0 % 16.0 % 8.0 % 5.0 % 7.0 % 3.0 % 1.0 % Iceland 334,311 47.6 % 26.4 % 9.3 % 1.6 % 8.4 % 4.6 % 1.7 % 0.4 % India 1,342,561,902 27.85 % 20.8 % 38.14 % 8.93 % 1.43 % 0.57 % 1.79 % 0.49 % Indonesia 263,519,317 36.82 % 25.87 % 28.85 % 7.96 % 0.18 % 0.13 % 0.15 % 0.04 % Iran 80,982,137 33.5 % 27.0 % 22.2 % 7.0 % 4.0 % 3.0 % 2.5 % 0.8 % Iraq 38,657,787 32.1 % 25.0 % 25.6 % 7.4 % 3.6 % 2.7 % 2.7 % 0.9 % Ireland 4,749,263 47.0 % 26.0 % 9.0 % 2.0 % 8.0 % 5.0 % 2.0 % 1.0 % Israel 8,323,659 32.0 % 34.0 % 17.0 % 7.0 % 3.0 % 4.0 % 2.0 % 1.0 % Italy 59,797,969 39.0 % 36.0 % 7.5 % 2.5 % 7.0 % 6.0 % 1.5 % 0.5 % Jamaica 2,813,316 47.0 % 23.0 % 20.0 % 3.0 % 3.5 % 2.0 % 1.0 % 0.5 % Japan 126,044,340 29.9 % 39.8 % 19.9 % 9.9 % 0.15 % 0.2 % 0.1 % 0.05 % Kenya 48,470,780 45.6 % 25.2 % 21.28 % 4.2 % 1.8 % 1.0 % 0.9 % 0.02 % Latvia 1,944,122 30.6 % 31.0 % 17.0 % 6.0 % 5.4 % 6.0 % 3.0 % 1.0 % Lebanon 6,039,438 38.4 % 32.3 % 9.5 % 3.2 % 7.7 % 6.5 % 1.7 % 0.7 % Libya 6,408,990 42.64 % 28.86 % 11.19 % 4.5 % 6.26 % 4.24 % 1.64 % 0.67 % Lithuania 2,827,573 36.0 % 33.0 % 11.0 % 4.0 % 7.0 % 6.0 % 2.0 % 1.0 % Luxemburg 587,297 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Macao 606,413 41.5 % 26.1 % 25.4 % 6.3 % 0.33 % 0.17 % 0.15 % 0.05 % Macedonia 2,083,690 30.0 % 34.0 % 15.0 % 6.0 % 5.0 % 6.0 % 3.0 % 1.0 % Malaysia 31,165,480 34.32 % 30.35 % 27.37 % 7.46 % 0.17 % 0.15 % 0.14 % 0.04 % Mauritania 4,266,765 46.3 % 26.68 % 17.47 % 3.85 % 2.8 % 1.6 % 1.1 % 0.2 % Mauritius 1,267,303 38.3 % 26.0 % 25.0 % 6.7 % 1.7 % 1.0 % 1.0 % 0.3 % Mexico 130,227,836 55.79 % 29.9 % 8.0 % 1.63 % 2.7 % 1.5 % 0.4 % 0.08 % Mongolia 3,052,044 55.4 % 27.7 % 10.9 % 4.96 % 0.6 % 0.3 % 0.1 % 0.04 % Myanmar 53,507,932 35.7 % 23.8 % 32.7 % 6.95 % 0.3 % 0.2 % 0.3 % 0.05 % Nepal 29,188,100 35.2 % 28.3 % 27.1 % 8.6 % 0.3 % 0.2 % 0.2 % 0.1 % Netherlands 17,033,012 39.5 % 35.0 % 6.7 % 2.5 % 7.5 % 7.0 % 1.3 % 0.5 % New Zealand 4,604,996 38.0 % 32.0 % 9.0 % 3.0 % 9.0 % 6.0 % 2.0 % 1.0 % Nigeria 191,851,411 51.3 % 22.4 % 20.7 % 2.6 % 1.6 % 0.7 % 0.6 % 0.1 % Norway 5,330,986 33.0 % 41.5 % 6.8 % 3.4 % 6.0 % 7.5 % 1.2 % 0.6 % North Korea 25,432,033 27.15 % 31.08 % 30.15 % 11.32 % 0.08 % 0.1 % 0.1 % 0.03 % Pakistan 180,440,005 24.63 % 20.6 % 34.40 % 9.52 % 4.17 % 2.66 % 3.57 % 0.45 % Papua New Guinea 7,952,511 48.0 % 37.2 % 9.6 % 2.8 % 1.0 % 0.8 % 0.4 % 0.2 % Peru 32,167,717 70.0 % 18.4 % 7.8 % 1.6 % 1.4 % 0.5 % 0.28 % 0.02 % Philippines 103,801,747 36.9 % 28.9 % 26.9 % 6.97 % 0.1 % 0.1 % 0.1 % 0.03 % Poland 38,563,480 31.0 % 32.0 % 15.0 % 7.0 % 6.0 % 6.0 % 2.0 % 1.0 % Portugal 10,264,672 36.3 % 40.0 % 6.6 % 2.9 % 6.0 % 6.6 % 1.1 % 0.5 % Republic of Moldova 4,043,722 28.5 % 31.8 % 17.6 % 7.0 % 5.0 % 6.0 % 3.0 % 1.1 % Romania 19,237,087 28.0 % 37.0 % 14.0 % 7.0 % 5.0 % 6.0 % 2.0 % 1.0 % Russia 143,374,801 28.0 % 30.0 % 20.0 % 7.0 % 4.9 % 5.8 % 3.2 % 1.1 % Saudi Arabia 32,744,532 47.8 % 23.9 % 17.0 % 4.0 % 4.0 % 2.0 % 1.0 % 0.3 % Serbia 8,776,827 31.92 % 35.28 % 12.6 % 4.2 % 6.08 % 6.72 % 2.4 % 0.8 % Singapore 5,784,819 43.6 % 23.9 % 24.4 % 6.0 % 0.9 % 0.7 % 0.4 % 0.1 % Slovakia 5,432,265 27.2 % 35.7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0 % 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 % 7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1 % 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.8 % 4.97 % 0.2 % 0.1 % 0.2 % 0.03 % Turkey 80,420,065 29.8 % 37.8 % 14.2 % 7.2 % 3.9 % 4.7 % 1.6 % 0.8 % Uganda 43,276,492 43.7 % 39.0 % 10.7 % 3.9 % 1.3 % 1.0 % 0.3 % 0.1 % Ukraine 44,404,078 32.0 % 34.0 % 15.0 % 5.0 % 5.0 % 6.0 % 2.0 % 1.0 % United Arab Emirates 3,398,017 44.1 % 21.9 % 20.9 % 4.3 % 4.3 % 2.1 % 2.0 % 0.4 % United Kingdom 65,512,375 37.0 % 35.0 % 8.0 % 3.0 % 7.0 % 7.0 % 2.0 % 1.0 % United States 326,481,533 37.4 % 35.7 % 8.5 % 3.4 % 6.6 % 6.3 % 1.5 % 0.6 % Venezuela 32,011,082 58.5 % 28.3 % 5.3 % 1.8 % 3.5 % 1.7 % 0.7 % 0.2 % Vietnam 95,414,640 41.7 % 21.9 % 30.8 % 4.98 % 0.3 % 0.1 % 0.2 % 0.02 % Zimbabwe 16,642,880 63.3 % 18.5 % 14.6 % 0.99 % 1.7 % 0.5 % 0.4 % 0.01 % World 6,635,281,257 38.67 % 27.42 % 22.02 % 5.88 % 2.55 % 1.99 % 1.11 % 0.36 %
  Fact 0.0: The table gives ABO and Rh blood type distribution by country as population averages.
  Fact 0.1: Each row lists a country, its population, and the percentages of O+, A+, B+, AB+, O−, A−, B−, and AB− blood types.
  Fact 0.2: Argentina has a population of 44,270,440, with blood type percentages O+ 45.4, A+ 34.26, B+ 8.59, AB+ 2.64, O− 8.4, A− 0.44, B− 0.21, AB− 0.06.
  Fact 0.3: Armenia has a population of 2,931,568, with blood type percentages O+ 29.0, A+ 46.3, B+ 12.0, AB+ 5.6, O− 2.0, A− 3.7, B− 1.0, AB− 0.4.
  Fact 0.4: Australia has a population of 24,642,693, with blood type percentages O+ 40.0, A+ 31.0, B+ 8.0, AB+ 2.0, O− 9.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.5: Austria has a population of 8,592,470, with blood type percentages O+ 30.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 8.0, B− 3.0, AB− 1.0.
  Fact 0.6: Bahrain has a population of 1,418,695, with blood type percentages O+ 48.48, A+ 19.35, B+ 22.61, AB+ 3.67, O− 3.27, A− 1.33, B− 1.04, AB− 0.25.
  Fact 0.7: Bangladesh has a population of 164,833,667, with blood type percentages O+ 31.18, A+ 21.44, B+ 34.58, AB+ 8.85, O− 1.39, A− 0.96, B− 0.96, AB− 0.64.
  Fact 0.8: Belgium has a population of 11,444,053, with blood type percentages O+ 38.0, A+ 34.0, B+ 8.6, AB+ 4.1, O− 7.0, A− 6.0, B− 1.5, AB− 0.8.
  Fact 0.9: Bolivia has a population of 11,053,376, with blood type percentages O+ 51.53, A+ 29.45, B+ 10.11, AB+ 1.15, O− 4.39, A− 2.73, B− 0.54, AB− 0.1.
  Fact 0.10: Bosnia and Herzegovina has a population of 3,792,730, with blood type percentages O+ 31.0, A+ 36.0, B+ 12.0, AB+ 6.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.11: Brazil has a population of 211,248,418, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.12: Bulgaria has a population of 7,045,097, with blood type percentages O+ 28.0, A+ 37.0, B+ 13.0, AB+ 7.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.13: Cambodia has a population of 16,077,172, with blood type percentages O+ 46.7, A+ 27.2, B+ 18.5, AB+ 4.9, O− 1.3, A− 0.8, B− 0.5, AB− 0.1.
  Fact 0.14: Cameroon has a population of 24,515,533, with blood type percentages O+ 42.8, A+ 38.8, B+ 12.0, AB+ 3.3, O− 1.4, A− 1.2, B− 0.4, AB− 0.1.
  Fact 0.15: Canada has a population of 36,627,140, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.6, AB+ 2.5, O− 7.0, A− 6.0, B− 1.4, AB− 0.5.
  Fact 0.16: Chile has a population of 18,314,060, with blood type percentages O+ 85.5, A+ 8.7, B+ 3.35, AB+ 1.0, O− 1.2, A− 0.1, B− 0.05, AB− 0.1.
  Fact 0.17: China has a population of 1,388,251,023, with blood type percentages O+ 47.7, A+ 27.8, B+ 18.9, AB+ 5.0, O− 0.28, A− 0.19, B− 0.1, AB− 0.03.
  Fact 0.18: Colombia has a population of 49,069,267, with blood type percentages O+ 61.3, A+ 26.11, B+ 2.28, AB+ 1.47, O− 5.13, A− 2.7, B− 0.7, AB− 0.31.
  Fact 0.19: Côte d'Ivoire has a population of 23,869,656, with blood type percentages O+ 46.5, A+ 22.5, B+ 22.5, AB+ 4.3, O− 2.0, A− 1.0, B− 1.0, AB− 0.2.
  Fact 0.20: Croatia has a population of 4,207,355, with blood type percentages O+ 29.0, A+ 36.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.21: Cuba has a population of 11,486,750, with blood type percentages O+ 45.8, A+ 33.5, B+ 10.2, AB+ 2.9, O− 3.6, A− 2.8, B− 1.0, AB− 0.2.
  Fact 0.22: Cyprus has a population of 1,189,395, with blood type percentages O+ 35.22, A+ 40.35, B+ 11.11, AB+ 4.72, O− 3.85, A− 3.48, B− 0.87, AB− 0.40.
  Fact 0.23: Czech Republic has a population of 10,555,152, with blood type percentages O+ 27.0, A+ 36.0, B+ 15.0, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.24: Denmark has a population of 5,711,902, with blood type percentages O+ 35.0, A+ 37.0, B+ 8.0, AB+ 4.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.25: Dominican Republic has a population of 10,766,932, with blood type percentages O+ 46.2, A+ 26.4, B+ 16.9, AB+ 3.1, O− 3.7, A− 2.1, B− 1.4, AB− 0.2.
  Fact 0.26: Ecuador has a population of 16,665,756, with blood type percentages O+ 75.0, A+ 14.0, B+ 7.1, AB+ 0.5, O− 2.38, A− 0.7, B− 0.3, AB− 0.02.
  Fact 0.27: Egypt has a population of 95,220,838, with blood type percentages O+ 52.0, A+ 24.0, B+ 12.4, AB+ 3.8, O− 5.0, A− 2.0, B− 0.6, AB− 0.2.
  Fact 0.28: El Salvador has a population of 6,171,483, with blood type percentages O+ 62.0, A+ 23.0, B+ 11.0, AB+ 1.0, O− 1.0, A− 1.0, B− 0.7, AB− 0.3.
  Fact 0.29: Estonia has a population of 1,305,745, with blood type percentages O+ 29.5, A+ 30.8, B+ 20.7, AB+ 6.3, O− 4.3, A− 4.5, B− 3.0, AB− 0.9.
  Fact 0.30: Ethiopia has a population of 104,352,727, with blood type percentages O+ 39.0, A+ 28.0, B+ 21.0, AB+ 5.0, O− 3.0, A− 2.0, B− 1.0, AB− 1.0.
  Fact 0.31: Finland has a population of 5,541,328, with blood type percentages O+ 27.0, A+ 38.0, B+ 15.0, AB+ 7.0, O− 4.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.32: France has a population of 64,939,560, with blood type percentages O+ 36.0, A+ 37.0, B+ 9.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.33: Germany has a population of 80,635,980, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.34: Greece has a population of 10,892,849, with blood type percentages O+ 37.4, A+ 32.9, B+ 11.0, AB+ 3.7, O− 7.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.35: Guinea has a population of 13,291,741, with blood type percentages O+ 46.88, A+ 21.64, B+ 22.86, AB+ 4.52, O− 2.0, A− 0.9, B− 1.0, AB− 0.2.
  Fact 0.36: Honduras has a population of 9,348,898, with blood type percentages O+ 57.5, A+ 27.0, B+ 7.8, AB+ 2.5, O− 2.7, A− 1.7, B− 0.6, AB− 0.2.
  Fact 0.37: Hong Kong has a population of 7,402,115, with blood type percentages O+ 41.5, A+ 26.13, B+ 25.34, AB+ 6.35, O− 0.32, A− 0.17, B− 0.14, AB− 0.05.
  Fact 0.38: Hungary has a population of 9,787,801, with blood type percentages O+ 27.0, A+ 33.0, B+ 16.0, AB+ 8.0, O− 5.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.39: Iceland has a population of 334,311, with blood type percentages O+ 47.6, A+ 26.4, B+ 9.3, AB+ 1.6, O− 8.4, A− 4.6, B− 1.7, AB− 0.4.
  Fact 0.40: India has a population of 1,342,561,902, with blood type percentages O+ 27.85, A+ 20.8, B+ 38.14, AB+ 8.93, O− 1.43, A− 0.57, B− 1.79, AB− 0.49.
  Fact 0.41: Indonesia has a population of 263,519,317, with blood type percentages O+ 36.82, A+ 25.87, B+ 28.85, AB+ 7.96, O− 0.18, A− 0.13, B− 0.15, AB− 0.04.
  Fact 0.42: Iran has a population of 80,982,137, with blood type percentages O+ 33.5, A+ 27.0, B+ 22.2, AB+ 7.0, O− 4.0, A− 3.0, B− 2.5, AB− 0.8.
  Fact 0.43: Iraq has a population of 38,657,787, with blood type percentages O+ 32.1, A+ 25.0, B+ 25.6, AB+ 7.4, O− 3.6, A− 2.7, B− 2.7, AB− 0.9.
  Fact 0.44: Ireland has a population of 4,749,263, with blood type percentages O+ 47.0, A+ 26.0, B+ 9.0, AB+ 2.0, O− 8.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.45: Israel has a population of 8,323,659, with blood type percentages O+ 32.0, A+ 34.0, B+ 17.0, AB+ 7.0, O− 3.0, A− 4.0, B− 2.0, AB− 1.0.
  Fact 0.46: Italy has a population of 59,797,969, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.5, AB+ 2.5, O− 7.0, A− 6.0, B− 1.5, AB− 0.5.
  Fact 0.47: Jamaica has a population of 2,813,316, with blood type percentages O+ 47.0, A+ 23.0, B+ 20.0, AB+ 3.0, O− 3.5, A− 2.0, B− 1.0, AB− 0.5.
  Fact 0.48: Japan has a population of 126,044,340, with blood type percentages O+ 29.9, A+ 39.8, B+ 19.9, AB+ 9.9, O− 0.15, A− 0.2, B− 0.1, AB− 0.05.
  Fact 0.49: Kenya has a population of 48,470,780, with blood type percentages O+ 45.6, A+ 25.2, B+ 21.28, AB+ 4.2, O− 1.8, A− 1.0, B− 0.9, AB− 0.02.
  Fact 0.50: Latvia has a population of 1,944,122, with blood type percentages O+ 30.6, A+ 31.0, B+ 17.0, AB+ 6.0, O− 5.4, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.51: Lebanon has a population of 6,039,438, with blood type percentages O+ 38.4, A+ 32.3, B+ 9.5, AB+ 3.2, O− 7.7, A− 6.5, B− 1.7, AB− 0.7.
  Fact 0.52: Libya has a population of 6,408,990, with blood type percentages O+ 42.64, A+ 28.86, B+ 11.19, AB+ 4.5, O− 6.26, A− 4.24, B− 1.64, AB− 0.67.
  Fact 0.53: Lithuania has a population of 2,827,573, with blood type percentages O+ 36.0, A+ 33.0, B+ 11.0, AB+ 4.0, O− 7.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.54: Luxemburg has a population of 587,297, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.55: Macao has a population of 606,413, with blood type percentages O+ 41.5, A+ 26.1, B+ 25.4, AB+ 6.3, O− 0.33, A− 0.17, B− 0.15, AB− 0.05.
  Fact 0.56: Macedonia has a population of 2,083,690, with blood type percentages O+ 30.0, A+ 34.0, B+ 15.0, AB+ 6.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.57: Malaysia has a population of 31,165,480, with blood type percentages O+ 34.32, A+ 30.35, B+ 27.37, AB+ 7.46, O− 0.17, A− 0.15, B− 0.14, AB− 0.04.
  Fact 0.58: Mauritania has a population of 4,266,765, with blood type percentages O+ 46.3, A+ 26.68, B+ 17.47, AB+ 3.85, O− 2.8, A− 1.6, B− 1.1, AB− 0.2.
  Fact 0.59: Mauritius has a population of 1,267,303, with blood type percentages O+ 38.3, A+ 26.0, B+ 25.0, AB+ 6.7, O− 1.7, A− 1.0, B− 1.0, AB− 0.3.
  Fact 0.60: Mexico has a population of 130,227,836, with blood type percentages O+ 55.79, A+ 29.9, B+ 8.0, AB+ 1.63, O− 2.7, A− 1.5, B− 0.4, AB− 0.08.
  Fact 0.61: Mongolia has a population of 3,052,044, with blood type percentages O+ 55.4, A+ 27.7, B+ 10.9, AB+ 4.96, O− 0.6, A− 0.3, B− 0.1, AB− 0.04.
  Fact 0.62: Myanmar has a population of 53,507,932, with blood type percentages O+ 35.7, A+ 23.8, B+ 32.7, AB+ 6.95, O− 0.3, A− 0.2, B− 0.3, AB− 0.05.
  Fact 0.63: Nepal has a population of 29,188,100, with blood type percentages O+ 35.2, A+ 28.3, B+ 27.1, AB+ 8.6, O− 0.3, A− 0.2, B− 0.2, AB− 0.1.
  Fact 0.64: Netherlands has a population of 17,033,012, with blood type percentages O+ 39.5, A+ 35.0, B+ 6.7, AB+ 2.5, O− 7.5, A− 7.0, B− 1.3, AB− 0.5.
  Fact 0.65: New Zealand has a population of 4,604,996, with blood type percentages O+ 38.0, A+ 32.0, B+ 9.0, AB+ 3.0, O− 9.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.66: Nigeria has a population of 191,851,411, with blood type percentages O+ 51.3, A+ 22.4, B+ 20.7, AB+ 2.6, O− 1.6, A− 0.7, B− 0.6, AB− 0.1.
  Fact 0.67: Norway has a population of 5,330,986, with blood type percentages O+ 33.0, A+ 41.5, B+ 6.8, AB+ 3.4, O− 6.0, A− 7.5, B− 1.2, AB− 0.6.
  Fact 0.68: North Korea has a population of 25,432,033, with blood type percentages O+ 27.15, A+ 31.08, B+ 30.15, AB+ 11.32, O− 0.08, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.69: Pakistan has a population of 180,440,005, with blood type percentages O+ 24.63, A+ 20.6, B+ 34.40, AB+ 9.52, O− 4.17, A− 2.66, B− 3.57, AB− 0.45.
  Fact 0.70: Papua New Guinea has a population of 7,952,511, with blood type percentages O+ 48.0, A+ 37.2, B+ 9.6, AB+ 2.8, O− 1.0, A− 0.8, B− 0.4, AB− 0.2.
  Fact 0.71: Peru has a population of 32,167,717, with blood type percentages O+ 70.0, A+ 18.4, B+ 7.8, AB+ 1.6, O− 1.4, A− 0.5, B− 0.28, AB− 0.02.
  Fact 0.72: Philippines has a population of 103,801,747, with blood type percentages O+ 36.9, A+ 28.9, B+ 26.9, AB+ 6.97, O− 0.1, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.73: Poland has a population of 38,563,480, with blood type percentages O+ 31.0, A+ 32.0, B+ 15.0, AB+ 7.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.74: Portugal has a population of 10,264,672, with blood type percentages O+ 36.3, A+ 40.0, B+ 6.6, AB+ 2.9, O− 6.0, A− 6.6, B− 1.1, AB− 0.5.
  Fact 0.75: Republic of Moldova has a population of 4,043,722, with blood type percentages O+ 28.5, A+ 31.8, B+ 17.6, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.1.
  Fact 0.76: Romania has a population of 19,237,087, with blood type percentages O+ 28.0, A+ 37.0, B+ 14.0, AB+ 7.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.77: Russia has a population of 143,374,801, with blood type percentages O+ 28.0, A+ 30.0, B+ 20.0, AB+ 7.0, O− 4.9, A− 5.8, B− 3.2, AB− 1.1.
  Fact 0.78: Saudi Arabia has a population of 32,744,532, with blood type percentages O+ 47.8, A+ 23.9, B+ 17.0, AB+ 4.0, O− 4.0, A− 2.0, B− 1.0, AB− 0.3.
  Fact 0.79: Serbia has a population of 8,776,827, with blood type percentages O+ 31.92, A+ 35.28, B+ 12.6, AB+ 4.2, O− 6.08, A− 6.72, B− 2.4, AB− 0.8.
  Fact 0.80: Singapore has a population of 5,784,819, with blood type percentages O+ 43.6, A+ 23.9, B+ 24.4, AB+ 6.0, O− 0.9, A− 0.7, B− 0.4, AB− 0.1.
  Fact 0.81: Slovakia has a population of 5,432,265, with blood type percentages O+ 27.2, A+ 35.7, B+ 15.3, AB+ 6.8, O− 4.8, A− 6.3, B− 2.7, AB− 1.2.
  Fact 0.82: Slovenia has a population of 2,071,258, with blood type percentages O+ 31.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.83: South Africa has a population of 55,437,815, with blood type percentages O+ 39.0, A+ 32.0, B+ 12.0, AB+ 3.0, O− 6.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.84: South Korea has a population of 50,748,307, with blood type percentages O+ 27.9, A+ 33.87, B+ 26.92, AB+ 10.98, O− 0.1, A− 0.13, B− 0.08, AB− 0.02.
  Fact 0.85: Spain has a population of 46,070,163, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.86: Sudan has a population of 40,827,975, with blood type percentages O+ 48.0, A+ 27.7, B+ 15.2, AB+ 2.8, O− 3.5, A− 1.8, B− 0.8, AB− 0.2.
  Fact 0.87: Sweden has a population of 9,920,843, with blood type percentages O+ 32.0, A+ 37.0, B+ 10.0, AB+ 5.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.88: Switzerland has a population of 8,454,321, with blood type percentages O+ 35.0, A+ 40.0, B+ 7.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.89: Syria has a population of 18,920,698, with blood type percentages O+ 43.0, A+ 30.0, B+ 14.0, AB+ 3.7, O− 5.0, A− 3.0, B− 1.0, AB− 0.3.
  Fact 0.90: Taiwan has a population of 23,234,936, with blood type percentages O+ 43.9, A+ 25.9, B+ 23.9, AB+ 6.0, O− 0.17, A− 0.1, B− 0.01, AB− 0.02.
  Fact 0.91: Thailand has a population of 68,298,027, with blood type percentages O+ 40.8, A+ 16.9, B+ 36.8, AB+ 4.97, O− 0.2, A− 0.1, B− 0.2, AB− 0.03.
  Fact 0.92: Turkey has a population of 80,420,065, with blood type percentages O+ 29.8, A+ 37.8, B+ 14.2, AB+ 7.2, O− 3.9, A− 4.7, B− 1.6, AB− 0.8.
  Fact 0.93: Uganda has a population of 43,276,492, with blood type percentages O+ 43.7, A+ 39.0, B+ 10.7, AB+ 3.9, O− 1.3, A− 1.0, B− 0.3, AB− 0.1.
  Fact 0.94: Ukraine has a population of 44,404,078, with blood type percentages O+ 32.0, A+ 34.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.95: United Arab Emirates has a population of 3,398,017, with blood type percentages O+ 44.1, A+ 21.9, B+ 20.9, AB+ 4.3, O− 4.3, A− 2.1, B− 2.0, AB− 0.4.
  Fact 0.96: United Kingdom has a population of 65,512,375, with blood type percentages O+ 37.0, A+ 35.0, B+ 8.0, AB+ 3.0, O− 7.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.97: United States has a population of 326,481,533, with blood type percentages O+ 37.4, A+ 35.7, B+ 8.5, AB+ 3.4, O− 6.6, A− 6.3, B− 1.5, AB− 0.6.
  Fact 0.98: Venezuela has a population of 32,011,082, with blood type percentages O+ 58.5, A+ 28.3, B+ 5.3, AB+ 1.8, O− 3.5, A− 1.7, B− 0.7, AB− 0.2.
  Fact 0.99: Vietnam has a population of 95,414,640, with blood type percentages O+ 41.7, A+ 21.9, B+ 30.8, AB+ 4.98, O− 0.3, A− 0.1, B− 0.2, AB− 0.02.
  Fact 0.100: Zimbabwe has a population of 16,642,880, with blood type percentages O+ 63.3, A+ 18.5, B+ 14.6, AB+ 0.99, O− 1.7, A− 0.5, B− 0.4, AB− 0.01.
  Fact 0.101: The World row has a population of 6,635,281,257, with blood type percentages O+ 38.67, A+ 27.42, B+ 22.02, AB+ 5.88, O− 2.55, A− 1.99, B− 1.11, AB− 0.36.

Retrieved items:
<item 1>
The table lists estimated population and percentages for eight blood types across many countries, plus a separate set of ethnic-group figures. The first listed
type is generally most common, especially in parts of Latin America (Chile 85.5%, Ecuador 75%, Peru 70%); B types are comparatively common in South and East
Asia. The final four types are usually rare. Patterns vary widely by country and ancestry.
</item 1>

<item 2>
Blood type distribution by country
Jump up ^ `` Portuguese Blood Institute '' ( in Portuguese ) . ( assuming Rh and AB antigens are independent ) Jump up ^ Sanguina Jump up ^ BASHAWRI , Layla A.
; AL - MULHIMiz , Abdul Az ; AHMAD , Malik S. ( 2001 ) . `` Frequency of ABO blood groups in the eastern region of Saudi Arabia '' . Saudi Medical Journal . 22
( 11 ) : 1008 -- 1101 . ISSN 0379 - 5284 . PMID 11744976 . Jump up ^ `` Najčastejšie otázky '' ( Frequently Asked Questions ) ( in Slovak ) . Národná transfúzna
služba SR . Retrieved 2017 - 09 - 22 . Jump up ^ `` What 's Your Type ? '' . South African National Blood Service . Archived from the original on 2010 - 02 - 06
. Retrieved 2010 - 11 - 19 .
</item 2>

<item 3>
Blood type distribution by country
Blood group A is associated with high frequencies in Europe , especially in Scandinavia and Central Europe , although its highest frequencies occur in some
Australian Aboriginal populations and the Blackfoot Indians of Montana , the US . Maps of allele ABO among native populations ( edit ) Map of allele A among
native populations Map of allele B among native populations Map of allele O among native populations References ( edit ) Jump up ^ CIA World Factbook Jump up ^
`` Study of Blood Immunogenetical Markers in Armenian Population '' ( PDF ) . 1989 . Jump up ^ Blood Types - What Are They ? , Australian Red Cross Jump up ^ ``
Austrian Red Cross - Blood Donor Information '' . Old.roteskreuz.at. 2006 - 03 - 21 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Rode Kruis Wielsbeke - Blood Donor
information material '' . Rodekruiswielsbeke.be . Archived from the original on 26 November 2010 . Retrieved 2010 - 11 - 19 .
</item 3>

<item 4>
Blood type distribution by country
Retrieved 2010 - 11 - 19 . Jump up ^ `` Federación Nacional de Donantes de Sangre / La sangre / Grupos '' . Donantesdesangre.net . Retrieved 2010 - 11 - 19 .
Jump up ^ `` Frequency of major blood groups in the Swedish population '' . Geblod.nu. 2007 - 10 - 02 . Archived from the original on 24 November 2010 .
Retrieved 2010 - 11 - 19 . Jump up ^ - BLUTSPENDE https://www.blutspende.ch/fr/don_de_sang/sanguin_groupe/groupe_sanguin - BLUTSPENDE Check url = value ( help )
. Missing or empty title = ( help ) Jump up ^ `` Turkey Blood Group Site '' . Kangrubu.com . Archived from the original on 29 May 2010 . Retrieved 2010 - 11 -
19 . Jump up ^ `` Blood Group Basics '' . Blood.co.uk . Retrieved 2017 - 07 - 30 .
</item 4>

<item 5>
Blood type distribution by country
7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0
% 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0
% 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.
</item 5>

<item 6>
Blood type distribution by country
In countries such as Austria , Denmark , Norway , and Switzerland , about 45 -- 50 % of the population have this blood type , whereas about 40 % of Poles and
Ukrainians do so . The highest frequencies are found in small , unrelated populations . For example , about 80 % of the Blackfoot Indians of Montana have blood
type A ... Jump up ^ Technical Monograph No. 2 : The ABO Blood Group System and ABO Subgroups ( PDF ) . Biotec . March 2005 . Archived from the original ( PDF )
on 6 February 2007 ... The frequency of blood group A is quite high ( 25 -- 55 % ) in Europe , especially in Scandinavia and parts of central Europe . High
group A frequency is also found in the Aboriginal people of South Australia ( up to 45 % ) and in certain American Indian tribes where the frequency reaches 35
% ...
</item 6>

<item 7>
Blood type distribution by country
2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia
104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 %
37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.
</item 7>

<item 8>
Blood type distribution by country
Retrieved 2017 - 11 - 23 . Jump up ^ `` Veregruppidest '' ( Blood groups ) . verekeskus.ee ( in Estonian ) . Retrieved 2017 - 11 - 23 . Jump up ^ ``
Suomalaisten veriryhmäjakauma '' ( in Finnish ) . 2009 - 08 - 21 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Les groupes sanguins ( système ABO ) '' . Centre
Hospitalier Princesse GRACE - Monaco ( in French ) . C.H.P.G. MONACO. 2005 . Retrieved 2008 - 07 - 15 . Jump up ^ `` Blóðflokkar '' ( in Icelandic ) ...
landspitali.is . Archived from the original on 2011 - 07 - 19 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Blood Type Frequencies by Country including the Rh
Factor '' .
</item 8>

<item 9>
Blood type distribution by country
975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 %
7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1
% 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.
</item 9>

<item 10>
Blood type distribution by country
40 11 Spanish 38 47 10 5 Sudanese 62 16 21 0 Swedish 38 47 10 5 Swiss 40 50 7 Tatars 28 30 29 13 Thais 37 22 33 8 Turks 43 34 18 6 Ukrainians 37 40 18 6 Black
Americans 49 27 20 White Americans 45 40 11 Vietnamese 42 22 30 5 Mean 43.91 34.80 16.55 5.14 Standard deviation 16.87 13.80 9.97 3.41 Blood group B has its
highest frequency in South Asia where it ranks first as the largest share of the earth 's population . In Southeast Asia the share of the population is high ,
especially in Thailand and Indonesia , then in East Asia , Northern Asia and neighboring Central Asia , and its incidence diminishes both towards the east and
the west , falling to single digit percentages in Netherlands , Norway , Portugal and Switzerland . It is believed to have been entirely absent from Native
American and Australian Aboriginal populations prior to the arrival of Europeans in those areas .
</item 10>
=== END ITEM id=ee5795dc3160be8f ===
