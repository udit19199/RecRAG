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

=== ITEM id=62e01bca820537ef ===
Gold sentences and facts:
Sentence 0: Important production centres today are the Hallertau in Germany ( more hop - growing area than any other country as of 2006 ) , the Yakima ( Washington ) and Willamette ( Oregon ) valleys , and western Canyon County , Idaho ( including the communities of Parma , Wilder , Greenleaf , and Notus ) . The principal production centres in the UK are in Kent ( which produces Kent Goldings hops ) , Herefordshire , and Worcestershire . Essentially all of the harvested hops are used in beer making .
  Fact 0.0: The Hallertau in Germany has more hop-growing area than any other country as of 2006.
  Fact 0.1: The Hallertau in Germany is an important hop production centre today.
  Fact 0.2: The Yakima valley in Washington is an important hop production centre today.
  Fact 0.3: The Willamette valley in Oregon is an important hop production centre today.
  Fact 0.4: Western Canyon County, Idaho is an important hop production centre today.
  Fact 0.5: Western Canyon County, Idaho includes the communities of Parma, Wilder, Greenleaf, and Notus.
  Fact 0.6: The principal hop production centres in the UK are in Kent.
  Fact 0.7: Kent produces Kent Goldings hops.
  Fact 0.8: The principal hop production centres in the UK are in Herefordshire.
  Fact 0.9: The principal hop production centres in the UK are in Worcestershire.
  Fact 0.10: Essentially all of the harvested hops are used in beer making.
Sentence 1: Hops production is concentrated in moist temperate climates , with much of the world 's production occurring near the 48th parallel north . Hop plants prefer the same soils as potatoes and the leading potato - growing states in the United States are also major hops - producing areas ; however , not all potato - growing areas can produce good hops naturally : soils in the Maritime Provinces of Canada , for example , lack the boron that hops prefer . Historically , hops were not grown in Ireland , but were imported from England . In 1752 more than 500 tons of English hops were imported through Dublin alone .
  Fact 1.0: Hops production is concentrated in moist temperate climates.
  Fact 1.1: Much of the world's hops production occurs near the 48th parallel north.
  Fact 1.2: Hop plants prefer the same soils as potatoes.
  Fact 1.3: The leading potato-growing states in the United States are also major hops-producing areas.
  Fact 1.4: Not all potato-growing areas can produce good hops naturally.
  Fact 1.5: Soils in the Maritime Provinces of Canada lack the boron that hops prefer.
  Fact 1.6: Historically, hops were not grown in Ireland.
  Fact 1.7: Historically, hops were imported into Ireland from England.
  Fact 1.8: In 1752 more than 500 tons of English hops were imported through Dublin alone.

Retrieved items:
<item 1>
Hops
Because pollinated seeds are undesirable for brewing beer , only female plants are grown in hop fields , thus preventing pollination . Female plants are
propagated vegetatively , and male plants are culled if plants are grown from seeds . Hop plants are planted in rows about 2 to 2.5 metres ( 7 to 8 ft ) apart .
Each spring , the roots send forth new bines that are started up strings from the ground to an overhead trellis . The cones grow high on the bine , and in the
past , these cones were picked by hand . Harvesting of hops became much more efficient with the invention of the mechanical hops separator , patented by Emil
Clemens Horst in 1909 . Harvest comes near the end of summer when the bines are pulled down and the flowers are taken to a hop house or oast house for drying .
Hop houses are two - story buildings , of which the upper story has a slatted floor covered with burlap . Here the flowers are poured out and raked even . A
heating unit on the lower floor is used to dry the hops .
</item 1>

<item 2>
Hops
Problems with powdery mildew and downy mildew devastated New York 's production by the 1920s , and California only produces hops on a small scale . Hop bars
were used before modern machinery was invented to make the holes for the hop poles . World production ( edit ) Hops production is concentrated in moist
temperate climates , with much of the world 's production occurring near the 48th parallel north . Hop plants prefer the same soils as potatoes and the leading
potato - growing states in the United States are also major hops - producing areas ; however , not all potato - growing areas can produce good hops naturally :
soils in the Maritime Provinces of Canada , for example , lack the boron that hops prefer . Historically , hops were not grown in Ireland , but were imported
from England . In 1752 more than 500 tons of English hops were imported through Dublin alone .
</item 2>

<item 3>
Hops
Hops - wikipedia Hops Jump to : navigation , search For other uses , see hop ( disambiguation ) . Hop flower in a hop yard in the Hallertau , Germany
Cross-section drawing of a hop Hops are the flowers ( also called seed cones or strobiles ) of the hop plant Humulus lupulus . They are used primarily as a
flavouring and stability agent in beer , to which they impart bitter , zesty , or citric flavours ; though they are also used for various purposes in other
beverages and herbal medicine . The hop plant is a vigorous , climbing , herbaceous perennial , usually trained to grow up strings in a field called a hopfield
, hop garden ( nomenclature in the South of England ) , or hop yard ( in the West Country and U.S. ) when grown commercially . Many different varieties of hops
are grown by farmers around the world , with different types used for particular styles of beer . The first documented use of hops in beer is from the 9th
century , though Hildegard of Bingen , 300 years later , is often cited as the earliest documented source .
</item 3>

<item 4>
Hops
Hops were then grown as far north as Aberdeen , near breweries for infrastructure convenience . According to Thomas Tusser 's 1557 Five Hundred Points of Good
Husbandry : `` The hop for his profit I thus do exalt , It strengtheneth drink and it flavoureth malt ; And being well - brewed long kept it will last , And
drawing abide , if ye draw not too fast . '' In England there were many complaints over the quality of imported hops , the sacks of which were often
contaminated by stalks , sand or straw to increase their weight . As a result , in 1603 , King James I approved an Act of Parliament banning the practice by
which `` the Subjects of this Realm have been of late years abused &c. to the Value of £ 20,000 yearly , besides the Danger of their Healths '' . Hop
cultivation was begun in the present - day United States in 1629 by English and Dutch farmers . Before prohibition , cultivation was mainly centred around New
York , California , Oregon , and Washington state .
</item 4>

<item 5>
Hops
private and public breeding programs are using their stock material . Particular hop varieties are associated with beer regions and styles , for example pale
lagers are usually brewed with European ( often German , Polish or Czech ) noble hop varieties such as Saaz , Hallertau and Strissel Spalt . British ales use
hop varieties such as Fuggles , Goldings and W.G.V. North American beers often use Cascade hops , Columbus hops , Centennial hops , Willamette , Amarillo hops
and about forty more varieties as the U.S.A. have lately been the more significant breeders of new hop varieties , including dwarf hop varieties . Hops from New
Zealand , such as Pacific Gem , Motueka and Nelson Sauvin , are used in a `` Pacific Pale Ale '' style of beer with increasing production in 2014 . Noble hops (
edit ) Mature hops growing in a hop yard in Germany The term `` noble hops '' traditionally refers to varieties of hops which are low in bitterness and high in
aroma . They are the European cultivars or races Hallertau , Tettnanger , Spalt , and Saaz .
</item 5>

<item 6>
Hops
In 1752 more than 500 tons of English hops were imported through Dublin alone . Important production centres today are the Hallertau in Germany ( more hop -
growing area than any other country as of 2006 ) , the Yakima ( Washington ) and Willamette ( Oregon ) valleys , and western Canyon County , Idaho ( including
the communities of Parma , Wilder , Greenleaf , and Notus ) . The principal production centres in the UK are in Kent ( which produces Kent Goldings hops ) ,
Herefordshire , and Worcestershire . Essentially all of the harvested hops are used in beer making .
</item 6>

<item 7>
Hops
Data may be shared with other brewers via BeerXML allowing the reproduction of a recipe allowing for differences in hop availability . Varieties ( edit ) Main
article : List of hop varieties This section needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( July 2014 ) ( Learn how and when to remove this template message ) Breeding programs ( edit )
There are many different varieties of hops used in brewing today . Historically , hops varieties were identified by geography ( such as Hallertau , Spalt , and
Tettnang from Germany ) , by the farmer who is recognized as first cultivating them ( such as Goldings or Fuggles from England ) , or by their growing habit (
e.g. , Oregon Cluster ) . Around 1900 , a number of institutions began to experiment with breeding specific hop varieties . The breeding program at Wye College
in Wye , Kent was started in 1904 and rose to prominence through the work of Prof. E.S. Salmon .
</item 7>

<item 8>
Hops
Contents ( hide ) 1 History 2 World production 2.1 Cultivation and harvest 2.2 Migrant labor and social impact 3 Chemical composition 3.1 Alpha acids 3.2 Beta
acids 3.3 Essential oils 3.4 Flavonoids 4 Brewing 5 Varieties 5.1 Breeding programs 5.2 Noble hops 6 Other uses 7 Toxicity 8 See also 9 References 10 External
links History ( edit ) The first documented hop cultivation was in 736 , in the Hallertau region of present - day Germany , although the first mention of the
use of hops in brewing in that country was 1079 . However , in a will of Pepin the Short , the father of Charlemagne , hop gardens were left to the Cloister of
Saint - Denis in 768 . Not until the 13th century did hops begin to start threatening the use of gruit for flavouring . Gruit was used when taxes were levied by
the nobility on hops . Whichever was taxed made the brewer then quickly switch to the other .
</item 8>

<item 9>
Hops
Essentially all of the harvested hops are used in beer making . Early season hop growth in a hop yard in the Yakima River Valley of Washington with Mount Adams
in the distance Hop producing country 2012 hop output in tonnes ( t ) Germany 34,434 United States 27,782 China 11,000 Czech Republic 4,338 Poland 2,206 North
Korea 2,000 Albania 1,650 United Kingdom 1,650 Slovenia 1,338 Cultivation and harvest ( edit ) A superstructure of overhead wires supports strings that in turn
support bines Although hops are grown in most of the continental United States and Canada , cultivation of hops for commercial production requires a particular
environment . As hops are a climbing plant , they are trained to grow up trellises made from strings or wires that support the plants and allow them
significantly greater growth with the same sunlight profile . In this way , energy that would have been required to build structural cells is also freed for
crop growth . The hop plant 's reproduction method is that male and female flowers develop on separate plants , although occasionally a fertile individual will
develop which contains both male and female flowers .
</item 9>

<item 10>
Hops
A heating unit on the lower floor is used to dry the hops . When dry , the hops are moved to a press , a sturdy box with a plunger . Two long pieces of burlap
are laid into the hop press at right angles , the hops are poured in and compressed into bales . Hop cones contain different oils , such as lupulin , a
yellowish , waxy substance , an oleoresin , that imparts flavour and aroma to beer . Lupulin contains lupulone and humulone , which possess antibiotic
properties , suppressing bacterial growth favoring brewer 's yeast to grow . After lupulin has been extracted in the brewing process the papery cones are
discarded . Migrant labor and social impact ( edit ) Hops harvest in the Kingdom of Bohemia ( 1898 ) Hops harvest in Skåne , Sweden in 1937 The need for massed
labor at harvest time meant hop - growing had a big social impact . Around the world , the labor - intensive harvesting work involved large numbers of migrant
workers who would travel for the annual hop harvest .
</item 10>
=== END ITEM id=62e01bca820537ef ===

=== ITEM id=779f4ea0ed138fbb ===
Gold sentences and facts:
Sentence 0: The term `` spirit '' in reference to alcohol stems from Middle Eastern alchemy . These alchemists were more concerned with medical elixirs than with transmuting lead into gold . The vapor given off and collected during an alchemical process ( as with distillation of alcohol ) was called a spirit of the original material .
  Fact 0.0: The term spirit used for alcohol stems from Middle Eastern alchemy.
  Fact 0.1: Middle Eastern alchemists were more concerned with medical elixirs than with transmuting lead into gold.
  Fact 0.2: The vapor given off and collected during an alchemical process was called a spirit of the original material.
  Fact 0.3: Distillation of alcohol is an example of such an alchemical process.

Retrieved items:
<item 1>
Distilled beverage
slivovitz 80 % -- 20 ° C ( 68 ° F ) 90 % or more -- 17 ° C ( 63 ° F ) -- neutral grain spirit Serving ( edit ) See also : Bartending terminology A row of
alcoholic beverages -- in this case , spirits -- in a bar Distilled beverages can be served : Neat -- at room temperature without any additional ingredient ( s
) Up -- shaken or stirred with ice , strained , and served in a stemmed glass . Down -- shaken or stirred with ice , strained , and served in a rocks glass . On
the rocks -- over ice cubes Blended or frozen -- blended with ice With a simple mixer , such as club soda , tonic water , juice , or cola As an ingredient of a
cocktail As an ingredient of a shooter With water With water poured over sugar ( as with absinthe ) Alcohol consumption by country ( edit ) Main article : List
of countries by alcohol consumption The World Health Organization measures and publishes alcohol consumption patterns in different countries . The WHO measures
alcohol consumed by persons 15 years of age or older and reports it on the basis of liters of pure alcohol consumed per capita in a given year in a country .
</item 1>

<item 2>
Distilled beverage
These all have a relatively low alcohol content , typically less than 15 % . Brandy is a spirit produced by the distillation of wine , and has an ABV of over 35
% . Other examples of distilled beverages include vodka , baijiu , gin , rum , tequila , mezcal , and whisky . ( Also see list of alcoholic drinks , and liquors
by national origin . ) Contents ( hide ) 1 Nomenclature 2 Etymology 3 History of distillation 3.1 Precursors 3.2 True distillation 3.3 Government regulation 3.4
Microdistilling 4 Flammability 5 Serving 6 Alcohol consumption by country 7 Health effects 7.1 Short - term effects 7.2 Long - term effects 8 See also 9
References 10 Bibliography 11 External links Nomenclature ( edit ) The term spirit refers to a distilled beverage that contains no added sugar and has at least
20 % alcohol by volume ( ABV ) . Distilled beverages bottled with added sugar and added flavorings , such as Grand Marnier , Frangelico , and American schnapps
, are known instead as liqueurs .
</item 2>

<item 3>
Distilled beverage
Consult your doctor on the benefits and risks of consuming alcohol in moderation . '' See also ( edit ) Liquor portal Drink portal Absinthe Akvavit Aguardiente
Alcoholic beverage Arak Arrack Awamori Baijiu / Shōchū / Soju Brandy Borovička Cachaça Eau de vie Er guo tou Fenny Freeze distillation Gin ( and Jenever )
Horilka Liquor store List of beverages Mamajuana Mezcal Moonshine Neutral grain spirit Orujo Pálinka Pisco Poitín Rakı Rakia Rum Rượu đế Schnapps Slivovitz
Vodka Whisky Tsikoudia Tsipouro References ( edit ) Jump up ^ `` distilled spirit - alcoholic beverage '' . Encyclopædia Britannica . Jump up ^ E. Gildemeister
and Fr . Hoffman , translated by Edward Kremers ( 1913 ) . The Volatile Oils. 1 . New York : Wiley . p. 203 . Jump up ^ Bryan H. Bunch and Alexander Hellemans (
2004 ) . The History of Science and Technology .
</item 3>

<item 4>
Distilled beverage
Distilled beverages generally have an alcohol concentration higher than 30 % . Beer and wine , which are not distilled beverages , are limited to a maximum
alcohol content of about 20 % ABV , as most yeasts can not reproduce when the concentration of alcohol is above this level ; as a consequence , fermentation
ceases at that point . Etymology ( edit ) The origin of `` liquor '' and its close relative `` liquid '' was the Latin verb liquere , meaning `` to be fluid ''
. According to the Oxford English Dictionary , an early use of the word in the English language , meaning simply `` a liquid '' , can be dated to 1225 . The
first use the OED mentions of its meaning `` a liquid for drinking '' occurred in the 14th century . Its use as a term for `` an intoxicating alcoholic drink ''
appeared in the 16th century . The term `` spirit '' in reference to alcohol stems from Middle Eastern alchemy . These alchemists were more concerned with
medical elixirs than with transmuting lead into gold . The vapor given off and collected during an alchemical process ( as with distillation of alcohol ) was
called a spirit of the original material .
</item 4>

<item 5>
Distilled beverage
Retrieved 2016 - 07 - 10 . The earliest possible period seems to be the Eastern Han dynasty ... the most likely period for the beginning of true distillation of
spirits for drinking in China is during the Jin and Southern Song dynasties Jump up ^ Taylor , F. Sherwood ( 1945 ) . `` The Evolution of the Still '' . Annals
of Science . 5 ( 3 ) : 186 . doi : 10.1080 / 00033794500201451 . ISSN 0003 - 3790 . ^ Jump up to : Ahmad Y. al - Hassan ( 2001 ) , Science and Technology in
Islam : Technology and applied sciences , pages 65 - 69 , UNESCO Jump up ^ Hassan , Ahmad Y. `` Alcohol and the Distillation of Wine in Arabic Sources '' .
History of Science and Technology in Islam . Retrieved 2014 - 04 - 19 . Jump up ^ The Economist : `` Liquid fire - The Arabs discovered how to distil alcohol .
</item 5>

<item 6>
Distilled beverage
The History of Science and Technology . Houghton Mifflin Harcourt . p. 88 . ISBN 0 - 618 - 22123 - 9 . Jump up ^ Marcelin Berthelot Collection des anciens
alchimistes grecs ( 3 vol. , Paris , 1887 -- 1888 , p. 161 ) ^ Jump up to : Irfan Habib ( 2011 ) , Economic History of Medieval India , 1200 - 1500 , page 55 ,
Pearson Education ^ Jump up to : Haw , Stephen G. ( 2006 ) . `` Wine , women and poison '' . Marco Polo in China . Routledge . pp. 147 -- 148 . ISBN 978 - 1 -
134 - 27542 - 7 . Retrieved 2016 - 07 - 10 .
</item 6>

<item 7>
Distilled beverage
Thus began the `` national '' drinks of Europe : jenever ( Belgium and the Netherlands ) , gin ( England ) , Schnaps ( Germany ) , grappa ( Italy ) , borovička
( Slovakia ) , horilka ( Ukraine ) , akvavit / snaps ( Scandinavia ) , vodka ( Poland and Russia ) , ouzo ( Greece ) , rakia ( the Balkans ) , and poitín (
Ireland ) . The actual names emerged only in the 16th century , but the drinks were well known prior to then . Government regulation ( edit ) It is legal to
distill beverage alcohol as a hobby for personal use in some countries , including New Zealand and the Netherlands . In the United States , it is illegal to
distill beverage alcohol without a license . In some parts of the U.S. , it is also illegal to sell a still without a license . However , all states allow
unlicensed individuals to make their own beer , and some also allow unlicensed individuals to make their own wine ( although making beer and wine is also
prohibited in some local jurisdictions ) .
</item 7>

<item 8>
Distilled beverage
Distilled beverage - wikipedia Distilled beverage Jump to : navigation , search `` Liquor '' redirects here . For sweetened and flavored alcoholic beverages ,
see Liqueur . For the novel series , see Liquor ( novel series ) . An old whiskey still A display of various distilled beverages in a supermarket Some single -
drink distilled beverage bottles available in Germany A distilled beverage , spirit , liquor , hard liquor or hard alcohol is an alcoholic beverage produced by
distillation of liquid drinks made with grains , fruit , or vegetables that have already gone through alcoholic fermentation . The distillation process purifies
the liquid and removes diluting components like water , for the purpose of increasing its proportion of alcohol content ( commonly expressed as alcohol by
volume , ABV ) . As distilled beverages contain significantly more alcohol , they are considered `` harder '' -- in North America , the term hard liquor is used
to distinguish distilled beverages from undistilled ones . As examples , this term does not include beverages such as beer , wine , mead , sake , or cider , as
they are fermented but not distilled . These all have a relatively low alcohol content , typically less than 15 % .
</item 8>

<item 9>
Distilled beverage
Forbes , Robert ( 1997 ) . Short History of the Art of Distillation from the Beginnings up to the Death of Cellier Blumenthal . Brill Academic Publishers . ISBN
90 - 04 - 00617 - 6 . Multhauf , Robert ( 1993 ) . The Origins of Chemistry . Gordon & Breach Science Publishers . ISBN 2 - 88124 - 594 - 3 . External links (
edit ) Wikimedia Commons has media related to Distilled beverages . History and Taxonomy of Distilled Spirits ( hide ) Alcoholic drinks History and production
History of alcohol History of alcoholic beverages History of beer History of wine History of Bordeaux wine History of Champagne History of Chianti History of
French wine History of Portuguese wine History of Rioja wine History of Sherry History of South African wine Production Alcohol industry Brewing Distillation
Drinking establishments Fermentation Winemaking Alcoholic drinks Fermented drink Beer Cider Mead Rice wine Sake Wine Other fermented drinks Distilled drink
Brandy Gin Liqueur Rum Tequila Vodka Whisky Other distilled drinks Fortified wine Madeira wine Marsala wine Port wine Sherry Vermouth Other fortified wines
Distilled drinks by
</item 9>

<item 10>
Distilled beverage
The production method was written in code , suggesting that it was being kept secret . In 1437 , `` burned water '' ( brandy ) was mentioned in the records of
the County of Katzenelnbogen in Germany . It was served in a tall , narrow glass called a Goderulffe . Claims upon the origin of specific beverages are
controversial , often invoking national pride , but they are plausible after the 12th century AD , when Irish whiskey and German brandy became available . These
spirits would have had a much lower alcohol content ( about 40 % ABV ) than the alchemists ' pure distillations , and they were likely first thought of as
medicinal elixirs . Consumption of distilled beverages rose dramatically in Europe in and after the mid-14th century , when distilled liquors were commonly used
as remedies for the Black Death . Around 1400 , methods to distill spirits from wheat , barley , and rye beers , a cheaper option than grapes , were discovered
.
</item 10>
=== END ITEM id=779f4ea0ed138fbb ===

=== ITEM id=c0b684f6327c0bde ===
Gold sentences and facts:
Sentence 0: The video takes place with the band and the album 's two protagonists Christian ( Josh Boswell ) and Gloria ( Lisa Stelly ) taking refuge in a white room after robbing a bank . The police arrive outside the room and open fire through the window , to the couple 's terror . As bullets rain through the room , the band continues playing . Gloria picks up the phone and throws it into a fish tank . As the bullets continue to fly and tear apart the room , Christian and Gloria become calm and walk toward each other , unharmed by the bullets . They embrace and kiss as the room goes dark , recreating the 21st Century Breakdown cover art . As the song ends , the room lights up again , and they are still uninjured . After the room lights up , there are various shots of destruction in the room , including a shot of some of the writing on the walls . The writing includes an excerpt of the lyrics to `` 21 Guns '' , as well as those of the song `` See The Light '' which also appears on `` 21st Century Breakdown '' . The video may represent the meaning of the song in the 21st Century Breakdown story .
  Fact 0.0: The video features the band
  Fact 0.1: The video features the album's two protagonists, Christian and Gloria
  Fact 0.2: Christian is played by Josh Boswell
  Fact 0.3: Gloria is played by Lisa Stelly
  Fact 0.4: Christian and Gloria take refuge in a white room after robbing a bank
  Fact 0.5: The police arrive outside the room and open fire through the window
  Fact 0.6: The police fire to the couple's terror
  Fact 0.7: As bullets rain through the room, the band continues playing
  Fact 0.8: Gloria picks up the phone and throws it into a fish tank
  Fact 0.9: As the bullets continue to fly and tear apart the room, Christian and Gloria become calm
  Fact 0.10: Christian and Gloria walk toward each other, unharmed by the bullets
  Fact 0.11: Christian and Gloria embrace and kiss as the room goes dark
  Fact 0.12: The embrace and kiss recreate the 21st Century Breakdown cover art
  Fact 0.13: As the song ends, the room lights up again
  Fact 0.14: Christian and Gloria are still uninjured when the room lights up
  Fact 0.15: After the room lights up, there are various shots of destruction in the room
  Fact 0.16: The shots include a shot of some of the writing on the walls
  Fact 0.17: The writing includes an excerpt of the lyrics to 21 Guns
  Fact 0.18: The writing also includes lyrics of the song See The Light
  Fact 0.19: The song See The Light also appears on 21st Century Breakdown
  Fact 0.20: The video may represent the meaning of the song in the 21st Century Breakdown story

Retrieved items:
<item 1>
21 Guns (song)
British Phonographic Industry . Enter 21 Guns in the search field and then press Enter . Jump up ^ `` American single certifications -- Green Day -- 21 Guns ''
. Recording Industry Association of America . Retrieved October 13 , 2015 . If necessary , click Advanced , then click Format , then select Single , then click
SEARCH External links Lyrics of this song at MetroLyrics ( hide ) Green Day Billie Joe Armstrong Mike Dirnt Tré Cool Jason White John Kiffmeyer Studio albums 39
/ Smooth Kerplunk Dookie Insomniac Nimrod Warning American Idiot 21st Century Breakdown ¡ Uno ! ¡ Dos ! ¡ Tré ! Revolution Radio Live albums Live Tracks Tune In
, Tokyo ... Bullet in a Bible Awesome as Fuck Compilation albums 1,039 / Smoothed Out Slappy Hours International Superhits ! Shenanigans Green Day Demolicious
Greatest Hits : God 's Favorite Band EPs 1,000 Hours Slappy Sweet Children Last Night on Earth : Live in Tokyo Singles `` Longview '' `` Welcome to Paradise ''
`` Basket Case '' `` When I Come Around '' `` She '' `` J.A.R. '' `` Geek
</item 1>

<item 2>
21 Guns (song)
The video was released February 1 . The second version of the song with the cast of American Idiot features Stark Sands , John Gallagher Jr. , and Michael Esper
singing their respective solos instead of Armstrong . It was performed with Armstrong singing lead male vocals at the 52nd Grammy Awards . Live EP 21 Guns Live
EP EP by Green Day Released September 4 , 2009 Recorded July 25 , 2009 in Albany , New York ; July 28 , 2009 Madison Square Garden , New York Genre Pop punk ,
punk rock , alternative rock Length 17 : 52 Label Reprise Producer Green Day Green Day released a live EP for the song including live performances of `` Welcome
to Paradise '' , `` Brain Stew / Jaded '' and `` F.O.D. '' recorded live in Albany , New York and Madison Square Garden . This EP was released on the Australian
iTunes store , Amazon UK , and Napster . Track listing Original version : Digital download / Maxi CD No . Title Length 1 . `` 21 Guns '' 5 : 21 2 .
</item 2>

<item 3>
21 Guns (song)
GreenDay.com . November 13 , 2009 . Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` Press Release '' . GreenDay.com .
December 2 , 2009 . Archived from the original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ Officially released video Green Day -- 21 Guns ( Cast
Version ) on YouTube Jump up ^ `` Get The New 21 Guns EP In Itunes Now '' . GreenDay.com . February 2 , 2010 . Archived from the original on June 8 , 2011 .
Retrieved July 12 , 2010 . Jump up ^ `` Australian-charts.com -- Green Day -- 21 Guns '' . ARIA Top 50 Singles . Retrieved October 27 , 2017 . Jump up ^ ``
Austriancharts.at -- Green Day -- 21 Guns '' ( in German ) . Ö3 Austria Top 40 . Retrieved October 27 , 2017 .
</item 3>

<item 4>
21 Guns (song)
21 Guns ( song ) - wikipedia 21 Guns ( song ) Jump to : navigation , search `` 21 Guns '' Single by Green Day from the album 21st Century Breakdown Released May
25 , 2009 ( digital download ) July 14 , 2009 ( CD single ) Format Digital download , CD single , vinyl Recorded 2008 Genre Alternative rock Length 5 : 21 Label
Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie , Mike Dirnt , and Tré Cool Producer ( s ) Butch Vig and Green Day Green Day singles chronology ``
Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere '' ( 2009 ) `` Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere
'' ( 2009 ) Audio sample file help `` 21 Guns '' is a song by American punk rock band Green Day . It was released as the second single from their eighth album ,
21st Century Breakdown .
</item 4>

<item 5>
21 Guns (song)
Retrieved June 16 , 2012 . Jump up ^ `` Green Day -- 21 Guns ( CD ) at Discogs '' . Discogs . Retrieved June 16 , 2012 . Jump up ^ `` Transformers : Revenge of
the Fallen Soundtrack '' . digiRAMA . Archived from the original on June 29 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` MTV Games , Harmonix & Warner Bros.
/ Reprise Records Partner With Green Day to Bring Their Music Exclusively to Rock Band / / News / / Rock Band '' . Rockband.com . Retrieved August 21 , 2009 .
Jump up ^ `` SkinnyLove21 's Journal -- The Vampire Diaries Soundtrack , Season 1 '' . last.fm. April 16 , 2010 . Archived from the original on August 18 , 2010
. Retrieved July 12 , 2010 . Jump up ^ `` '' 21 Guns '' -- Green Day `` . rockband.com . Archived from the original on April 25 , 2010 .
</item 5>

<item 6>
21 Guns (song)
The video may represent the meaning of the song in the 21st Century Breakdown story . Musical cast version `` 21 Guns '' Single by Green Day and the cast of
American Idiot from the album American Idiot : The Original Broadway Cast Recording B - side `` Like a Rolling Stone ( Bob Dylan Cover ) '' Released December 3
, 2009 Format Digital download , CD single Recorded 2009 Genre Alternative rock Length 4 : 41 Label Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie
, Mike Dirnt , and Tré Cool Producer ( s ) Green Day Green Day and the cast of American Idiot singles chronology `` 21 Guns '' ( 2009 ) `` When It 's Time '' (
2010 ) `` 21 Guns ( Musical Cast Version ) '' ( 2010 ) `` When It 's Time '' ( 2010 ) Green Day has recorded two versions of `` 21 Guns '' with the cast of
American Idiot . The recording was produced by Billie Joe Armstrong . The recording with Armstrong singing the male solo parts was released on December 3 to
Spinner.com and also released for digital download on December 22 .
</item 6>

<item 7>
21 Guns (song)
rockband.com . Archived from the original on April 25 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Live EP '' . GreenDay.com . September 20 , 2009 .
Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Video '' . Green Day.com . June 5 , 2009 . Archived from the
original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ `` 2009 MTV Video Music Awards Winners '' . MTV . Retrieved July 12 , 2010 . Jump up ^ `` Green
Day -- making of 21 Guns video '' . MySpace Video . June 23 , 2009 . Archived from the original on July 2 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` Press
Release '' . GreenDay.com . November 13 , 2009 .
</item 7>

<item 8>
21 Guns (song)
It became available for download on the music video game series Rock Band on July 7 , 2009 , along with the songs `` East Jesus Nowhere '' and `` Know Your
Enemy . The song was featured in episode 6 of The Vampire Diaries . The song is featured in the video game Green Day : Rock Band and is downloadable content for
the Rock Band series . A live version was also released on 21 Guns Live EP in September 2009 . Music video A music video was directed by Marc Webb and filmed in
Los Angeles on June 6 , 2009 . It premiered on MySpace on Monday , June 22 , 2009 at midnight EDT . A shorter version of the video also exists , which is set to
the radio edit of the song . It is the third Green Day video to feature touring guitarist Jason White performing with the band , following `` Wake Me Up When
September Ends '' and `` Working Class Hero '' . He is also seen briefly in the music videos for `` When I Come Around '' and `` Jesus of Suburbia '' but he is
not performing with the band .
</item 8>

<item 9>
Graph query row:
person: Lisa Stelly
occupation: None
relationship: RELATED_TO
work: 21 Guns music video
</item 9>

<item 10>
21 Guns (song)
Frustrators The Lookouts The Big Cats The Influents Related articles Discography List of songs Awards and nominations Adeline Records Cigarettes and Valentines
Money Money 2020 Stop Drop and Roll ! ! ! American Edit Green Day 's American Idiot soundtrack Green Day : Rock Band songs ¡ Cuatro ! Foreverly Book Portal
Retrieved from `` https://en.wikipedia.org/w/index.php?title=21_Guns_(song)&oldid=820470590 '' Categories : 2000s ballads 2009 singles Green Day songs Rock
ballads Songs written by Billie Joe Armstrong Song recordings produced by Butch Vig Music videos directed by Marc Webb Songs written by Mike Dirnt 2008 songs
2009 songs Number - one singles in Poland Reprise Records singles Hidden categories : CS1 Italian - language sources ( it ) Wikipedia pages semi-protected from
banned users Use mdy dates from March 2012 Articles with hAudio microformats All articles with unsourced statements Articles with unsourced statements from
October 2010 Music infoboxes with deprecated parameters Singlechart usages for Australia Singlechart usages for Austria Singlechart usages for Flanders
Singlechart usages for
</item 10>
=== END ITEM id=c0b684f6327c0bde ===

=== ITEM id=aca752502a40694e ===
Gold sentences and facts:
Sentence 0: Driver Seasons Entries Podiums Percentage Michael Schumacher 1991 -- 2006 , 2010 -- 2012 308 155 7001503200000000000 ♠ 50.32 % Lewis Hamilton 2007 -- 2018 211 119 7001564000000000000 ♠ 56.40 % Alain Prost 1980 -- 1991 , 1993 202 106 7001524800000099999 ♠ 52.48 % Sebastian Vettel 2007 -- 2018 202 101 7001500000000000000 ♠ 50.00 % 5 Fernando Alonso 2001 , 2003 -- 2018 296 97 7001327700000000000 ♠ 32.77 % 6 Kimi Räikkönen 2001 -- 2009 , 2012 -- 2018 276 93 7001337000000000000 ♠ 33.70 % 7 Ayrton Senna 1984 -- 1994 162 80 7001493800000000000 ♠ 49.38 % 8 Rubens Barrichello 1993 -- 2011 326 68 7001208600000000000 ♠ 20.86 % 9 David Coulthard 1994 -- 2008 247 62 7001251000000000000 ♠ 25.10 % 10 Nelson Piquet 1978 -- 1991 207 60 7001289900000000000 ♠ 28.99 % Source :
  Fact 0.0: The table lists drivers with their seasons, entries, podiums, and percentage.
  Fact 0.1: Michael Schumacher raced in the seasons 1991 to 2006 and 2010 to 2012.
  Fact 0.2: Michael Schumacher had 308 entries.
  Fact 0.3: Michael Schumacher had 155 podiums.
  Fact 0.4: Michael Schumacher has a percentage of 50.32%.
  Fact 0.5: Lewis Hamilton raced in the seasons 2007 to 2018.
  Fact 0.6: Lewis Hamilton had 211 entries.
  Fact 0.7: Lewis Hamilton had 119 podiums.
  Fact 0.8: Lewis Hamilton has a percentage of 56.40%.
  Fact 0.9: Alain Prost raced in the seasons 1980 to 1991 and 1993.
  Fact 0.10: Alain Prost had 202 entries.
  Fact 0.11: Alain Prost had 106 podiums.
  Fact 0.12: Alain Prost has a percentage of 52.48%.
  Fact 0.13: Sebastian Vettel raced in the seasons 2007 to 2018.
  Fact 0.14: Sebastian Vettel had 202 entries.
  Fact 0.15: Sebastian Vettel had 101 podiums.
  Fact 0.16: Sebastian Vettel has a percentage of 50.00%.
  Fact 0.17: Fernando Alonso is ranked 5th in the table.
  Fact 0.18: Fernando Alonso raced in the seasons 2001 and 2003 to 2018.
  Fact 0.19: Fernando Alonso had 296 entries.
  Fact 0.20: Fernando Alonso had 97 podiums.
  Fact 0.21: Fernando Alonso has a percentage of 32.77%.
  Fact 0.22: Kimi Räikkönen is ranked 6th in the table.
  Fact 0.23: Kimi Räikkönen raced in the seasons 2001 to 2009 and 2012 to 2018.
  Fact 0.24: Kimi Räikkönen had 276 entries.
  Fact 0.25: Kimi Räikkönen had 93 podiums.
  Fact 0.26: Kimi Räikkönen has a percentage of 33.70%.
  Fact 0.27: Ayrton Senna is ranked 7th in the table.
  Fact 0.28: Ayrton Senna raced in the seasons 1984 to 1994.
  Fact 0.29: Ayrton Senna had 162 entries.
  Fact 0.30: Ayrton Senna had 80 podiums.
  Fact 0.31: Ayrton Senna has a percentage of 49.38%.
  Fact 0.32: Rubens Barrichello is ranked 8th in the table.
  Fact 0.33: Rubens Barrichello raced in the seasons 1993 to 2011.
  Fact 0.34: Rubens Barrichello had 326 entries.
  Fact 0.35: Rubens Barrichello had 68 podiums.
  Fact 0.36: Rubens Barrichello has a percentage of 20.86%.
  Fact 0.37: David Coulthard is ranked 9th in the table.
  Fact 0.38: David Coulthard raced in the seasons 1994 to 2008.
  Fact 0.39: David Coulthard had 247 entries.
  Fact 0.40: David Coulthard had 62 podiums.
  Fact 0.41: David Coulthard has a percentage of 25.10%.
  Fact 0.42: Nelson Piquet is ranked 10th in the table.
  Fact 0.43: Nelson Piquet raced in the seasons 1978 to 1991.
  Fact 0.44: Nelson Piquet had 207 entries.
  Fact 0.45: Nelson Piquet had 60 podiums.
  Fact 0.46: Nelson Piquet has a percentage of 28.99%.
  Fact 0.47: The table ends with a 'Source' label that has no source given.

Retrieved items:
<item 1>
List of Formula One driver records
1953 Indianapolis 500 Tony Brooks 1957 British Grand Prix 9 4th race Bob Sweikert 1955 Indianapolis 500 Jacques Villeneuve 1996 European Grand Prix Source :
Most races before first win ( edit ) Start Driver Race 130th race Mark Webber 2009 German Grand Prix 123rd race Rubens Barrichello 2000 German Grand Prix 119th
race Jarno Trulli 2004 Monaco Grand Prix 113th race Jenson Button 2006 Hungarian Grand Prix 5 111th race Nico Rosberg 2012 Chinese Grand Prix 6 110th race
Giancarlo Fisichella 2003 Brazilian Grand Prix 7 96th race Mika Häkkinen 1997 European Grand Prix 8 95th race Thierry Boutsen 1989 Canadian Grand Prix 9 91st
race Jean Alesi 1995 Canadian Grand Prix 10 82nd race Eddie Irvine 1999 Australian Grand Prix Valtteri Bottas 2017 Russian Grand Prix Sources : Most races
without a win ( edit ) Driver Entries Starts Best result Andrea de Cesaris 214 208 2nd Nick
</item 1>

<item 2>
The list catalogs Formula One driver records across starts, wins, points, podiums, poles, fastest laps, longevity, and race incidents. Lewis Hamilton leads
consecutive starts (211); Mark Webber needed the most starts for a first win (130); Andrea de Cesaris made the most entries without winning (214); Nico
Hülkenberg has the most starts without a podium (138). John Watson won from 22nd on the grid, Michael Schumacher won the French Grand Prix eight
</item 2>

<item 3>
The records cover Formula One entries, starts, constructor and engine partnerships, wins and podiums. Rubens Barrichello leads entries (326) and starts (322);
Jenson Button (309 entries) and Michael Schumacher (308) follow. Schumacher tops single-constructor appearances with Ferrari (181 entries), while Lewis Hamilton
leads single-engine records with Mercedes (211). Among non-champions, Stirling Moss has the most wins (16); Nick Heidfeld has the most podium
</item 3>

<item 4>
List of Formula One driver records
Piquet 1987 Monaco -- 1987 Portuguese Michael Schumacher 2000 -- 2001 2000 Hungarian -- 2001 Brazilian 2004 European -- 2004 Italian Lewis Hamilton 2007 2007
Australian -- 2007 British Nico Rosberg 2015 2015 Australian -- 2015 British 2016 2016 Belgian -- 2016 Abu Dhabi Source : Most consecutive podium finishes from
first race of season ( edit ) Driver Season Consecutive podium finishes Podiums Michael Schumacher 2002 Australian -- Japanese 17 Fernando Alonso 2006 Bahrain
-- Canadian 9 Lewis Hamilton 2007 Australian -- British 2015 Australian -- British Sebastian Vettel 2011 Australian -- British Nico Rosberg 2015 Australian --
British 7 Nico Rosberg 2014 Australian -- Austrian 8 8 Niki Lauda 1976 Brazilian -- Swedish 7 Alain Prost 1988 Brazilian -- French Michael Schumacher 1994
Brazilian -- French Jenson Button 2009 Australian -- Turkish Source needed Youngest Drivers to score a podium finish ( edit ) ( only the first podium finish for
each driver is listed ) Driver Age Place Race Max
</item 4>

<item 5>
List of Formula One driver records
Pastor Maldonado ( 2012 Spanish Grand Prix ) , Mark Webber ( 2012 Monaco Grand Prix ) , Lewis Hamilton ( 2012 Canadian Grand Prix ) Fewest race wins in World
Championship winning year Mike Hawthorn ( 1958 ) Keke Rosberg ( 1982 ) World Champion with fewest career wins Mike Hawthorn , Phil Hill Most race wins in one
season without becoming World Champion 10 Lewis Hamilton ( 2016 ) Podiums Highest percentage of podium finishes in one season 100 Michael Schumacher , 17 podium
finishes out of 17 races in 2002 Most podiums for the same team 116 Michael Schumacher ( Ferrari ) Longest time between successive podium finishes 2,870 days
Alexander Wurz ( 1997 British Grand Prix -- 2005 San Marino Grand Prix ) Most consecutive podium finishes ( starting from debut ) 9 Lewis Hamilton ( 2007
Australian Grand Prix -- 2007 British Grand Prix ) Most podium finishes before a victory 15 Jean Alesi , Mika Häkkinen ,
</item 5>

<item 6>
List of Formula One driver records
3 days Max Verstappen ( 2014 Japanese Grand Prix ) Finishes Most classified finishes 244 Jenson Button Most race finishes 235 Michael Schumacher Most
consecutive classified finishes 41 Nick Heidfeld ( 2007 French Grand Prix -- 2009 Italian Grand Prix ) Most consecutive race finishes 33 Nick Heidfeld ( 2007
Chinese Grand Prix -- 2009 Italian Grand Prix ) Most consecutive finishes from start of career 27 Esteban Ocon ( 2016 Belgian Grand Prix -- 2017 Mexican Grand
Prix ) Most retirements 148 Andrea de Cesaris Most consecutive retirements 22 ( 18 DNF ) Andrea de Cesaris Most retirements in a season 16 ( 14 DNF ) Andrea de
Cesaris ( from 16 races in 1987 ) Longest time between first and last points finishes 7,749 days Michael Schumacher ( 1991 Italian Grand Prix -- 2012 Brazilian
Grand Prix ) Longest time between successive points finishes 2,
</item 6>

<item 7>
List of Formula One driver records
positions Youngest Oldest Most races without Fastest laps Total Percentage In one season Highest percentage in one season Youngest Oldest Most at the same GP
Podium finishes Total Percentage In one season Consecutive Consecutive from first race of season Youngest Oldest Most career starts without a podium Most career
podiums without a win Most career podiums without a World Championship Most races before scoring a podium finish Points Career points Total races finished in
the points Consecutive races finished in the points Highest average points per race started Highest percentage of races finished in the points Most in one
season Youngest Oldest Most points without a win Most without being World Champion World Champions with fewest career points Most races before scoring points
Most races without scoring points Race leaders Every lap , total For at least one lap , total Youngest Most laps Longest distance Most laps without a win Most
consecutive laps in the lead Multiple driver records Pole and win in same race Most wins from pole position in a season Pole , win and fastest lap in same race
Pole , win , fastest lap ,
</item 7>

<item 8>
The excerpt highlights Formula One podium and finishing records. Michael Schumacher leads total podiums with 155, ahead of Lewis Hamilton (119), Alain Prost
(106), Sebastian Vettel (101) and Fernando Alonso (97). Schumacher also holds the longest podium streak (19); Hamilton has 16 and Alonso 15. Schumacher recorded
17 consecutive podiums in 2002. Jenson Button has the most classified finishes (244), Schumacher the most race
</item 8>

<item 9>
List of Formula One driver records
2004 -- 2017 269 11 41 7 Riccardo Patrese 1977 -- 1993 256 6 37 8 Jacques Laffite 1974 -- 1986 176 6 32 Jean Alesi 1989 -- 2001 201 10 Juan Pablo Montoya 2001
-- 2006 94 7 30 Source : Most races before scoring a podium finish ( edit ) Entry Driver Race 91st race Martin Brundle 1992 French Grand Prix 73rd race Mika
Salo 1999 German Grand Prix 68th race Jenson Button 2004 Malaysian Grand Prix 67th race Johnny Herbert 1995 Spanish Grand Prix Pedro de la Rosa 2006 Hungarian
Grand Prix 6 60th race Gianni Morbidelli 1995 Australian Grand Prix 7 57th race Felipe Massa 2006 European Grand Prix 8 56th race Mark Webber 2005 Monaco Grand
Prix 9 55th race Kamui Kobayashi 2012 Japanese Grand Prix Daniel Ricciardo 2014 Spanish Grand Prix Source : Points ( edit ) Throughout the history of the World
Championship ,
</item 9>

<item 10>
List of Formula One driver records
2009 -- 2011 81 80 6th 10 Marcus Ericsson 2014 -- 2018 79 79 8th Source : Most career podiums without a win ( edit ) Driver Entries Starts Podiums Nick Heidfeld
185 183 13 Stefan Johansson 103 79 12 Chris Amon 108 97 11 Romain Grosjean 127 125 10 5 Martin Brundle 165 158 9 Eddie Cheever 143 132 Jean Behra 53 52 8 Luigi
Villoresi 34 31 8 9 Sergio Pérez 140 137 7 10 Andrea de Cesaris 214 208 5 Source : Most career podiums without a World Championship ( edit ) Driver Seasons
Starts Wins Podiums Rubens Barrichello 1993 -- 2011 326 11 68 David Coulthard 1994 -- 2008 247 13 62 Gerhard Berger 1984 -- 1997 210 10 48 Carlos Reutemann 1972
-- 1982 146 12 45 5 Mark Webber 2002
</item 10>
=== END ITEM id=aca752502a40694e ===

=== ITEM id=f5cd6775fd67cc66 ===
Gold sentences and facts:
Sentence 0: In most of the world speed limit signs display the limitation within a red circle . This design follows the style set out by the Vienna Convention on Road Signs and Signals , with the exception of United States and Canada . All countries globally worldwide ( except the United Kingdom and United States ) use metric speed limits in kilometres per hour . Metric signs are optional in the United States though less commonly seen , while Samoa uses both units . In Ontario , Canada , the type , location , and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act .
  Fact 0.0: In most of the world, speed limit signs display the limitation within a red circle.
  Fact 0.1: The red-circle speed limit sign design follows the style set out by the Vienna Convention on Road Signs and Signals.
  Fact 0.2: The Vienna Convention style is not followed by the United States and Canada.
  Fact 0.3: All countries worldwide, except the United Kingdom and the United States, use metric speed limits in kilometres per hour.
  Fact 0.4: Metric speed limit signs are optional in the United States.
  Fact 0.5: Metric speed limit signs are less commonly seen in the United States.
  Fact 0.6: Samoa uses both units for speed limits.
  Fact 0.7: In Ontario, Canada, the type, location, and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act.

Retrieved items:
<item 1>
Speed limit
Speed limit - wikipedia Speed limit Jump to : navigation , search This article is about road speed limits . For speed of light , see special relativity . For
road speed limits in specific countries , see Speed limits by country . For rail speed limits , see Slow zone . A typical 60 km / h speed limit sign used in
Australia A typical 50 mph speed limit sign in the United States A typical 50 mph speed limit sign in the United Kingdom and 50 km / h speed limit sign in Hong
Kong Road speed limits are used in most countries to set the maximum ( or minimum in some cases ) speed at which road vehicles may legally travel on particular
stretches of road . Speed limits may be variable and in some places speed is unlimited ( e.g. on some Autobahn sections in Germany ) . Speed limits are normally
indicated on a traffic sign . Speed limits are commonly set by the legislative bodies of nations or provincial governments and enforced by national or regional
police or judicial bodies . The first maximum speed limit was the 10 mph ( 16 km / h ) limit introduced in the United Kingdom in 1861 .
</item 1>

<item 2>
Speed limit
Retrieved 2009 - 09 - 16 . ^ Jump up to : http://tti.tamu.edu/documents/1465-1.pdf Jump up ^
http://azdot.gov/business/engineering-and-construction/traffic/faq/establishing-speed-limits Jump up ^
http://www.ksdot.org/bureaus/burTrafficSaf/brochures/establishingspeedlimits.asp Jump up ^ Public opposition to speed limits being set by an authority , often
arise because such agency has been viewed as abusing its power -- such as by arbitrary indiscretion or by creating `` speed traps . '' Because an expert can
theoretically calculate a safer speed limit , than the populace 's vote by driving , it is beneficial that local governments preserve strong public trust with
their integrity in speed regulation . See A Policy on Geometric Design of Highways and Streets , AASHTO , 4th Ed. , 2001 ; ISBN 1 - 56051 - 156 - 7 Jump up ^
USLIMITS2 : A Tool to Aid Practitioners in Determining Appropriate Speed Limit Recommendations Jump up ^ `` Riggs v. Gasser Motors , 22 Cal . App .
</item 2>

<item 3>
Speed limit
Per distance traveled , consequences of inappropriate speed are more frequent on lower speed , lower quality roads ; in the United States , for example , the ``
speeding fatality rate for local roads is three times that for Interstates '' Citations for violations of the basic speed law without a crash have sometimes
been ruled unfairly vague or arbitrary , hence a violation the due process of law , at least in the State of Montana . Even within states , differing
jurisdictions ( counties and cities ) choose to prosecute similar cases with differing approaches . Maximum speed limits ( edit ) A speed limit sign in Ontario
In France city sign has value of speed limit A street limited at 30 km / h End of speed limit , previous speed limit applies Most public roads in most countries
have a legally assigned numerical maximum speed limit which applies on all roads unless otherwise stated ; lower speed limits are often shown on a sign at the
start of the restricted section , although the presence of streetlights or the physical arrangement of the road may sometimes also be used instead . A posted
speed limit may only apply to that road or to all roads beyond the sign that defines them depending on local laws .
</item 3>

<item 4>
Speed limit
In the European Union , large signposts showing the national ( default ) speed limits of the respective country are usually erected immediately after border
crossings , with a repeater sign some 200 to 500 m ( 660 to 1,640 ft ) after the first sign . Some places provide an additional `` speed zone ahead '' ahead of
the restriction and speed limit reminder signs may appear at regular intervals which may be painted on the road surface . Signs are normally placed on both
sides of the road and in some places there are small ( less than 1 / 4 the size of the sign ) rectangular orange reflector flags attached to both upper right
corners of both signs . The speed limit sign marking the new speed zone may also have the orange flags ; this practice can be observed in New York on highways
where the speed limit varies such as New York State Route 17 . Speed limits on European Motorways in km / h Signage in many countries , especially in Europe ,
conforms to the Vienna Convention on Road Signs and Signals using black text with a red circle on a white background . In the United States , the signs are
usually rectangular with the words `` SPEED LIMIT '' and the values in black on a white background .
</item 4>

<item 5>
Speed limit
p. 343 . External links ( edit ) Wikimedia Commons has media related to Speed limit signs and Driver speed and relative risk of an accident . Global map of
speed limits from Wikispeedia A Walk Through The History of Speed Limit Signs in The US John F. Carr 's State Traffic and Speed Laws Global map of speed limits
from OpenStreetMap Traffic signs Signs Warning sign Regulatory sign Prohibitory traffic sign Mandatory sign Special regulation sign Stop sign Yield sign
Crossbuck Dead End sign One - way traffic Speed limit ( by country ) Advisory speed limit Information sign Direction , position , or indication sign Variable -
message sign Bilingual sign Driver location signs Logo sign Priority to the right By comparison Comparison of European road signs Comparison of MUTCD -
influenced traffic signs Comparison of traffic signs in English - speaking countries By country Africa Botswana Mauritius South Africa Zimbabwe Asia Azerbaijan
Bangladesh Brunei Cambodia China Hong Kong India Indonesia Iran Israel Japan Laos Malaysia Nepal Pakistan Singapore South Korea Sri Lanka Thailand Turkey United
Arab Emirates Vietnam Europe Austria Bosnia and Herzegovina Czech Republic Denmark Estonia Finland France Germany Greece Iceland Ireland Italy Liechtenstein
Netherlands Norway Poland Sweden Switzerland Ukraine United Kingdom Scotland Wales North America Canada Mexico United States Oceania Australia New Zealand South
America Argentina Brazil Chile Colombia Lights Traffic
</item 5>

<item 6>
Speed limit
North America Canada Mexico United States Oceania Australia New Zealand South America Argentina Brazil Chile Colombia Lights Traffic light Traffic - light
signalling and operation Traffic light control and coordination Typefaces Alfabeto Normale ( IT ) Austria ( AT ) Caractères ( FR ) Clarendon ( US ) Clearview (
US / CA / ID / PH / LK ) DIN 1451 ( DE / SADC ) Drogowskaz ( PL ) Enigmatic ( JP / KR ) Highway Gothic ( Global ) Bitstream Vera Sans ( TW ) Frutiger ( Europe )
GOST 10807 - 78 ( USSR ) GOST R 52290 - 2004 ( RU ) Hangil ( KR ) LLM Lettering ( MY ) Motorway ( Europe ) NPS Rawlinson Roadway ( US ) Panno ( KR ) SNV ( CH )
Tern ( Europe ) Trafikkalfabetet ( NO ) Transport ( UK ) Tratex ( SE ) Vialog ( JP ) Conventions Geneva Convention on Road Traffic Vienna Convention on Road
Traffic Vienna Convention on Road Signs and Signals Manual on Uniform Traffic Control Devices ( US ) SADC Road Traffic Signs Manual ( SADC ) Traffic Signs
Regulations and General Directions ( UK )
</item 6>

<item 7>
Speed limit
Variable speed limits ( edit ) Example variable speed limit sign in the United States Digital speed limit sign for variable speed limits In 1965 , the first
known experiments with variable speed limit signs took place on a 30 km stretch of German motorway A8 between Munich and the border city of Salzburg , Austria .
Mechanically variable message signs could display speeds of 60 , 80 and 100 km / h , `` danger zone '' or `` accident '' . Personnel monitored traffic using
video technology , and manually controlled the signage . Beginning in the 1970s , additional advanced traffic control systems were put into service . Modern
motorway control systems can work without human intervention using various types of sensors to measure traffic flow and weather conditions . In 2009 , 1,300 km
( 810 mi ) of German motorways were equipped with such systems . In the late 1960s , heavily traveled portions of the New Jersey Turnpike began using variable
speed limit signs , in combination with variable message signs . Officials can adjust the speed limit according to weather , traffic conditions , and
construction .
</item 7>

<item 8>
Speed limit
The U.S. Federal Highway Administration 's Manual on Uniform Traffic Control Devices provides guidelines for the appearance of speed limit signs . In Alaska and
California , speed limits are often labeled `` MAXIMUM SPEED '' instead . In Oregon , most speed limit signs just say `` SPEED '' and the number . Canada has
similar signs bearing the legend `` MAXIMUM '' and in km / h instead of mph . `` Maximum '' is used instead of `` Speed Limit '' because it has similar meanings
in English and French . Australian signs are rectangular but have a red circle like the Vienna Convention signs . The speed limit is commonly set at or below
the 85th percentile operating speed ( being the speed which no more than 15 % of traffic is exceeding ) and in the US is frequently set 4 to 8 mph ( 6 to 13 km
/ h ) below that speed . Thus , if the 85th percentile operating speed as measured by a Traffic and Engineering Survey exceeds the design speed , legal
protection is given to motorists traveling at such speeds ( design speed is `` based on conservative assumptions about driver , vehicle and roadway
characteristics '' ) .
</item 8>

<item 9>
Speed limit
The speeding fatality rate for local roads is three times that for Interstates Jump up ^ http://www.pe.com/articles/speed-776756-survey-limit.html Jump up ^
http://caselaw.findlaw.com/mt-supreme-court/1110919.html Jump up ^
http://vincenzeslaw.com/4-ways-virginia-reckless-driving-charges-handled-differently-fairfax-county-arlington-alexandria-stafford/ Jump up ^ `` Traffic
Operations : Road Signs '' . Government of South Australia . 2002 - 08 - 23 ( last update date ) . Retrieved 2010 - 09 - 26 . Most regulatory signs are
rectangular with a black legend on a white background ... ( exceptions include ) speed restriction signs ( symbol enclosed in a red circle ) '' Check date
values in : date = ( help ) Jump up ^ `` Speed Zoning Information '' ( PDF ) . Institute of Transportation Engineers . 22 March 2004 . Retrieved 2009 - 09 - 17
. Jump up ^ MUTCD Sections 2B.
</item 9>

<item 10>
Speed limit
But `` → '' and `` ここ から '' is `` from here '' . Philippines Philippines End speed limit 35 mph United States End speed limit ( custom ) United States ( archaic
) Advisory speed limit ( edit ) Further information : Advisory speed limit Advisory speed limits may provide a safe suggested speed in an area , or warn of the
safe maximum speed for dangerous curves . See also ( edit ) Advisory speed limit Assured Clear Distance Ahead ( ACDA ) Design speed Functional classification
Negligence per se Operating speed Radar speed sign Solomon curve Speed limits by country Stopping sight distance Traffic violations reciprocity Traffic
psychology Notes ( edit ) Jump up ^ World Health Organisation ( 2004 ) p. foreword , `` In high - income countries , an established set of interventions have
contributed to significant reductions in the incidence and impact of road traffic injuries . These include the enforcement of legislation to control speed and
alcohol consumption , mandating the use of seat - belts and crash helmets , and the safer design and use of roads and vehicles . '' ^ Jump up to : Table 3 , J.
Stuster and Z .
</item 10>
=== END ITEM id=f5cd6775fd67cc66 ===

=== ITEM id=c26b93e95f1e9e44 ===
Gold sentences and facts:
Sentence 0: `` Like a Boss '' is a song written and recorded by comedy hip hop troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean version that was produced in both audio and video . In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work .
  Fact 0.0: Like a Boss is a song.
  Fact 0.1: Like a Boss was written and recorded by the comedy hip hop troupe The Lonely Island.
  Fact 0.2: The Lonely Island recorded Like a Boss for their debut studio album Incredibad.
  Fact 0.3: Like a Boss is a parody of the first official single from Slim Thug's first album Already Platinum.
  Fact 0.4: The first official single from Slim Thug's first album Already Platinum is also named Like a Boss.
  Fact 0.5: Like a Boss had multiple versions.
  Fact 0.6: One version of Like a Boss was audio-only and did not have the celebrity interviewer.
  Fact 0.7: Like a Boss had a clean version that was produced in both audio and video.
  Fact 0.8: In 2011 Like a Boss was subject to a lawsuit.
  Fact 0.9: The 2011 lawsuit over Like a Boss was from the producers of the song.
  Fact 0.10: The producers of Like a Boss claimed they never received proper compensation for their work.

Retrieved items:
<item 1>
Like a Boss
In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work . Contents ( hide ) 1
Song 2 Video 3 Popularity 4 Lawsuit 5 Censorship 6 References 7 External links Song ( edit ) `` Like a Boss '' premiered as a Saturday Night Live Digital Short
on April 4 , 2009 . In the song , Andy Samberg is a businessman , evidently in middle management , having a performance review . The reviewer asks him to
describe the things he does in an average day , and does so for the remainder of the song in his trademark aggressive style while repeating the lyric `` like a
boss '' after every activity . He first lists activities which are fairly mundane , such as communicating to bigger bosses in corporate headquarters , approving
memorandums , remembering birthdays , leading a workshop , and promoting synergy . However , the song takes a foul turn after he mentions his female subordinate
Deborah , whom he repeatedly pursues , rejecting him , causing him to `` swallow sadness '' , face a sexual harassment lawsuit and be passed up for a promotion
.
</item 1>

<item 2>
Like a Boss
Like a Boss - wikipedia Like a Boss Jump to : navigation , search ( hide ) This article has multiple issues . Please help improve it or discuss these issues on
the talk page . ( Learn how and when to remove these template messages ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page . ( January 2015 ) This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed .
</item 2>

<item 3>
Like a Boss
Abby Elliott and Bobby Moynihan can be seen in the video as background dancers . Popularity ( edit ) Prior to the release of the video on Saturday Night Live ,
the term `` Like a Boss '' was not often searched for or used in social context . Though it was indeed searched for prior to the release of the video , the term
clearly spiked at release and peaked on Google in April 2009 , hitting a 100 on the Google Trend chart . It has also become popular on social media sites such
as Twitter Tumblr , and Facebook , where it is a popular hashtag to cite a task completed , goal achieved or successful action that someone is proud of doing .
It has also led to parody videos such as `` Like a Bus '' , where a bus is shown flying through traffic and often bad weather conditions with no regard or
slowing for the bad conditions . On 22 October 2012 , `` Like A Boss '' surpassed 100 million views . The video was featured by The Huffington Post , and TV
Guide . Lawsuit ( edit ) In November 2011 a production team from St. Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic
Records .
</item 3>

<item 4>
Like a Boss
Unsourced material may be challenged and removed . ( February 2011 ) ( Learn how and when to remove this template message ) ( Learn how and when to remove this
template message ) `` Like a Boss '' Song by The Lonely Island featuring Seth Rogen from the album Incredibad Recorded 2008 Genre Comedy hip hop Length 1 : 46
Label Universal Republic Records Songwriter ( s ) Andy Samberg , Akiva Schaffer , Jorma Taccone , A. Banks Producer ( s ) Rick tha Rular Incredibad track
listing `` Shrooms ( skit ) '' ( 9 ) `` Like a Boss '' ( 10 ) `` We Like Sportz '' ( 11 ) `` Like a Boss '' is a song written and recorded by comedy hip hop
troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already
Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean
version that was produced in both audio and video .
</item 4>

<item 5>
Like a Boss
Banks , however , claims that he has yet to see any sort of a royalty for the song . Another track from the duo was later featured on the band 's 2011 album
Turtleneck & Chain in the song `` Shy Ronnie 2 : Ronnie & Clyde . '' Censorship ( edit ) This section does not cite any sources . Please help improve this
section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2015 ) ( Learn how and when to remove this
template message ) On the NBC version of this digital short , the part where Andy has a gun in his mouth and whines , `` Oh fuck , man , I ca n't fuckin ' do it
. Shit ! '' had the shot of Andy with the gun in his mouth shortened and redubbed the line to , `` Oh crap , man , I ca n't effin ' do it . Ugh ! ! ''
References ( edit ) Jump up ^ Van Dril , Eric ( 5 April 2009 ) . `` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' .
</item 5>

<item 6>
“Like a Boss” is a 2009 comedy-rap song by The Lonely Island, parodying Slim Thug’s song of the same name. Premiering as an SNL Digital Short, it depicts Andy
Samberg’s middle manager describing increasingly outrageous, often explicit daily activities, each done “like a boss.” The video features Seth Rogen and SNL
cast cameos. It became a popular online catchphrase and passed 100 million views in 2012. In 2011,
</item 6>

<item 7>
Like a Boss
`` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' . The Ripcord . Archived from the original on 11 April 2009 . Retrieved 2014 - 06 - 21 . Jump up ^
`` Google Trends - websearch interest : like a boss - Worldwide , 2004 - present '' . Google Trends . Jump up ^ `` Twitter / search - # likeaboss '' .
Twitter.com . Jump up ^ `` like a boss on Tumblr '' . Tumblr.com . Jump up ^ `` # likeaboss '' . Facebook.com . Jump up ^ `` Like a BUS ! Playlist '' .
Youtube.com . Jump up ^ `` Like A Boss : SNL Digital Short ( VIDEO ) '' . The Huffington Post . 6 May 2009 . Retrieved 2014 - 06 - 23 . Jump up ^ DiNunno , Gina
. `` Today 's News : Our Take - Seth Rogen and Andy Samberg in SNL Digital Short `` Like a Boss '' `` . TV Guide.com . Retrieved 23 June 2014 .
</item 7>

<item 8>
Like a Boss
`` Jack Sparrow '' `` 3 - Way ( The Golden Rule ) '' `` YOLO '' `` Spring Break Anthem '' `` Hugs '' Other songs `` Lazy Sunday '' `` Like a Boss '' `` Cool
Guys Do n't Look at Explosions '' `` Everything Is Awesome '' Film Hot Rod ( 2007 ) Popstar : Never Stop Never Stopping ( 2016 ) Television SNL Digital Shorts
Party Over Here Book Retrieved from `` https://en.wikipedia.org/w/index.php?title=Like_a_Boss&oldid=816720124 '' Categories : The Lonely Island songs 2009
singles Saturday Night Live sketches Comedy rap songs Hidden categories : Wikipedia introduction cleanup from January 2015 All pages needing cleanup Articles
covered by WikiProject Wikify from January 2015 All articles covered by WikiProject Wikify Articles needing additional references from February 2011 All
articles needing additional references Articles with multiple maintenance issues Articles with hAudio microformats Articles needing additional references from
January 2015 Talk Contents About Wikipedia Nederlands Simple English Edit links This page was last edited on 23 December 2017 , at 06 : 20 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 8>

<item 9>
Like a Boss
The video ends with the giant fish delivering the final words `` like a boss . '' Video ( edit ) The video features Seth Rogen , recording over the vocals done
by Akiva Schaffer on the single. The video adds additional visuals not included with the words of the song . It ends with Samberg 's tombstone being shown (
which has the epitaph `` He was da Best '' ) , and a skeptical Rogen trying to confirm some of the more bizarre events with Samberg . The `` bomb the Russians
'' part shows Samberg attacking St. Basil 's Cathedral . The video features brief cameo appearances from Saturday Night Live writer John Mulaney as the worker
that Samberg micromanages , Casey Wilson as the woman giving Samberg the memo , Kristen Wiig as Samberg 's office crush Debra , Jason Sudeikis as Samberg 's
boss and Bill Hader as the man on whom Samberg performs fellatio . It also features Jorma Taccone as the man giving him the lawsuit , and Akiva Schaffer selling
him the gun . Abby Elliott and Bobby Moynihan can be seen in the video as background dancers .
</item 9>

<item 10>
Like a Boss
TV Guide.com . Retrieved 23 June 2014 . Jump up ^ Kaufman , Gill . `` ' SNL , ' Andy Samberg Sued By Producers over ' Like A Boss ' - MTV '' . MTV News .
Retrieved 11 July 2014 . External links ( edit ) Beard , Lanford ( November 8 , 2011 ) . `` NBC , Andy Samberg sued for copyright infringement '' .
Entertainment Weekly . Retrieved November 8 , 2011 . Original video : `` Like a Boss '' on YouTube Censored video : `` Like a Boss - Clean Version '' on YouTube
Video and Song Lyrics : `` Like a Boss '' Lonely Island 's Song Page : `` Like a Boss '' ( hide ) The Lonely Island Andy Samberg Akiva Schaffer Jorma Taccone
Discography Studio albums Incredibad Turtleneck & Chain The Wack Album Popstar : Never Stop Never Stopping Singles `` Dick in a Box '' `` Jizz in My Pants '' ``
I 'm on a Boat '' `` I Just Had Sex '' `` The Creep '' `` We 're Back ! ''
</item 10>
=== END ITEM id=c26b93e95f1e9e44 ===
