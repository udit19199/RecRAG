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

=== ITEM id=8dcd20bf56209cfd ===
Gold sentences and facts:
Sentence 0: `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper , spending five weeks at the top .
  Fact 0.0: 'Sacrifice' is a ballad performed by musician Elton John.
  Fact 0.1: The lyrics of 'Sacrifice' are by Bernie Taupin.
  Fact 0.2: The music of 'Sacrifice' is by Elton John.
  Fact 0.3: 'Sacrifice' appears on the 1989 album Sleeping with the Past.
  Fact 0.4: 'Sacrifice' was first released in October 1989.
  Fact 0.5: 'Sacrifice' was released in 1990.
  Fact 0.6: 'Sacrifice' was the second single from the album Sleeping with the Past.
  Fact 0.7: 'Sacrifice' achieved success, particularly in France and the UK.
  Fact 0.8: 'Sacrifice' became Elton John's first solo chart-topper in France and the UK.
  Fact 0.9: 'Sacrifice' spent five weeks at the top of the chart.

Retrieved items:
<item 1>
Sacrifice (song)
Billboard Hot Adult Contemporary Tracks Chart ( 1990 ) Peak position Australian ARIA Singles Chart 7 Dutch Mega Top 100 Eurochart Hot 100 French SNEP Singles
Chart German Singles Chart 36 Irish Singles Chart New Zealand RIANZ Singles Chart 7 Norwegian Singles Chart Swiss Singles Chart 23 UK Singles Chart U.S.
Billboard Hot 100 18 Zimbabwe Singles Chart 6 Year - end charts ( edit ) Chart ( 1990 ) Rank Australian Singles Chart 47 Dutch Top 40 11 UK Singles Charts
Certifications and sales ( edit ) Country Certification Date Sales certified Physical sales France Silver 200,000 380,000 UK Platinum 1 September 1990 600,000
Personnel ( edit ) Elton John : vocals ,
</item 1>

<item 2>
Sacrifice (song)
Sacrifice ( song ) - wikipedia Sacrifice ( song ) Jump to : navigation , search For other songs with the same title , see Sacrifice ( disambiguation ) § Songs .
`` Sacrifice '' Single by Elton John from the album Sleeping with the Past A-side `` Healing Hands '' ( second release ) B - side `` Love Is a Cannibal '' (
first release ) Released October 1989 ( first release ) May 1990 ( second release ) Format CD , vinyl record ( 7 '' and 12 `` ) , audio cassette Length 5 : 11
Label Rocket , MCA Songwriter ( s ) Elton John , Bernie Taupin Producer ( s ) Chris Thomas Elton John singles chronology `` Healing Hands '' ( 1989 ) ``
Sacrifice '' ( 1989 ) `` Club at the End of the Street '' ( 1990 ) `` Healing Hands '' ( 1989 ) `` Sacrifice '' ( 1989 ) `` Club at the End of the Street '' (
1990 ) `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John .
</item 2>

<item 3>
Sacrifice (song)
Official Charts Company . Official Charts Company . Retrieved 20 January 2016 . ^ Jump up to : `` Elton John singles , Billboard charts '' . allmusic .
Retrieved 9 September 2008 . Jump up ^ `` '' Sacrifice `` , Australian Singles Chart '' . australian - charts . Retrieved 13 December 2007 . Jump up ^ `` ''
Sacrifice `` , Dutch Mega Top 100 '' ( in Dutch ) . dutchcharts . Retrieved 11 April 2010 . Jump up ^ `` '' Sacrifice `` , French Singles Chart '' ( in French )
. lescharts . Retrieved 13 December 2007 . Jump up ^ German Singles Chart Charts-surfer.de Archived 8 January 2009 at the Wayback Machine . ( Retrieved 9
September 2008 ) Jump up ^ `` Irish Single Chart , database '' . irishcharts . Retrieved 9 September 2008 . Jump up ^ `` '' Sacrifice '' New Zealand Singles
Chart `` . charts.org.nz . Retrieved 11 April 2010 .
</item 3>

<item 4>
Sacrifice (song)
Jump up ^ `` UK certifications , database '' . Bpi . Retrieved 9 September 2008 . External links ( edit ) Lyrics of this song at MetroLyrics Elton John songs
1960s and 1970s `` Skyline Pigeon '' ( 1968 ) `` I 've Been Loving You '' ( 1968 ) `` Lady Samantha '' ( 1969 ) `` It 's Me That You Need '' ( 1969 ) `` Border
Song '' ( 1970 ) `` Rock and Roll Madonna '' ( 1970 ) `` From Denver to L.A. '' ( 1970 ) `` Take Me to the Pilot '' ( 1970 ) `` Your Song '' ( 1970 ) `` Friends
'' ( 1971 ) `` Levon '' ( 1971 ) `` Tiny Dancer '' ( 1972 ) `` Rocket Man '' ( 1972 ) `` Honky Cat '' ( 1972 ) `` I Think I 'm Going to Kill Myself '' ( 1972 )
`` Mona Lisas and Mad Hatters '' ( 1972 ) `` Crocodile Rock '' ( 1972 ) `` Elderberry Wine '' (
</item 4>

<item 5>
Sacrifice (song)
The song is beloved by both John and Bernie Taupin and deemed a bookend of sorts to their first hit `` Your Song '' . Taupin has said he is surprised he wrote
it , and said in an interview , `` I think ' Sacrifice ' is one of the best songs we 've written . '' The song is not a typical love song , but rather a song
about a breakup of a marriage where the loss of the relationship is `` no sacrifice . '' Although the single was successful , Elton John did not tour Europe in
both 1989 -- 1990 to support the Sleeping with the Past album ( with the exception of one appearance in Knebworth Park on 30 June 1990 ) touring only in the
U.S. and Australia . The 1989 European tour was , instead , in support of the Reg Strikes Back album , with no material representing Sleeping with the Past . In
all shows that were done to support Sleeping with the Past in 1989 -- 1990 , Elton played this song solo on his Roland RD - 1000 Digital Piano .
</item 5>

<item 6>
Sacrifice (song)
Awarded as The Best Song Retrieved from `` https://en.wikipedia.org/w/index.php?title=Sacrifice_(song)&oldid=833955469 '' Categories : 1989 singles 1990 singles
Disney songs Elton John songs European Hot 100 Singles number - one singles SNEP Top Singles number - one singles UK Singles Chart number - one singles Songs
with lyrics by Bernie Taupin Songs with music by Elton John Pop ballads Rock ballads Song recordings produced by Chris Thomas ( record producer ) 1989 songs MCA
Records singles The Rocket Record Company singles Music videos directed by Alek Keshishian Hidden categories : CS1 Dutch - language sources ( nl ) CS1 French -
language sources ( fr ) Webarchive template wayback links CS1 Norwegian - language sources ( no ) CS1 German - language sources ( de ) Articles with hAudio
microformats Use dmy dates from August 2010 Talk Contents About Wikipedia Dansk Español Français Italiano Nederlands Norsk nynorsk Polski Português Edit links
This page was last edited on 3 April 2018 , at 05 : 22 .
</item 6>

<item 7>
Sacrifice (song)
Thus , it became John 's first solo number - one single in the UK Singles Chart , remaining on the top spot for five weeks . With this re-release , Elton John
also got his first number one in France and stayed on the chart for 26 weeks . Track listings ( edit ) First release 12 '' single -- UK `` Sacrifice '' -- 5 :
07 `` Love Is a Cannibal '' ( From the Columbia Motion Picture Ghostbusters II ) -- 3 : 53 CD single -- U.S. `` Sacrifice '' -- 5 : 07 `` Love Is a Cannibal ''
( from the Columbia Motion Picture Ghostbusters II ) -- 3 : 53 `` Durban Deep '' Second release UK single `` Sacrifice '' -- 5 : 07 `` Healing Hands '' -- 4 :
22 Charts and sales ( edit ) Weekly charts ( edit ) Chart ( 1989 ) Peak position UK Singles Chart 55 U.S.
</item 7>

<item 8>
Sacrifice (song)
000 Personnel ( edit ) Elton John : vocals , Roland RD - 1000 digital piano Davey Johnstone : electric guitar Romeo Williams : bass Jonathan Moffett : drums Guy
Babylon : keyboards Fred Mandel : keyboards Preceded by `` Words '' by The Christians French SNEP number - one single 26 May 1990 -- 9 June 1990 ( 3 weeks )
Succeeded by `` Bo le lavabo ( WC Kiss ) '' by Lagaf ' Preceded by `` Vogue '' by Madonna Eurochart Hot 100 number - one single 16 June 1990 -- 28 July 1990 ( 7
weeks ) Succeeded by `` U Ca n't Touch This '' by MC Hammer Preceded by `` World In Motion '' by Englandneworder UK number - one single 17 June 1990 -- 21 July
1990 ( 5 weeks ) Succeeded by `` Turtle Power '' by Partners in Kryme References ( edit ) ^ Jump up to : `` Elton John - Full Official Chart History '' .
Official Charts Company . Official Charts Company . Retrieved 20 January 2016 .
</item 8>

<item 9>
Sacrifice (song)
The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then
in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper ,
spending five weeks at the top . A country music cover of the song was performed by Don Henley and Vince Gill as part of the 2018 tribute album Restoration :
Reimagining the Songs of Elton John and Bernie Taupin . Contents ( hide ) 1 Overview 1.1 Background and writing 1.2 Music video 1.3 Chart performance 2 Track
listings 3 Charts and sales 3.1 Weekly charts 3.2 Year - end charts 3.3 Certifications and sales 4 Personnel 5 References 6 External links Overview ( edit )
Background and writing ( edit ) `` Sacrifice '' is inspired both by the Aretha Franklin song `` Do Right Woman , Do Right Man '' and described as a Percy Sledge
song by Elton John .
</item 9>

<item 10>
Sacrifice (song)
'' ( 1997 ) `` Something About the Way You Look Tonight '' ( 1997 ) `` Candle in the Wind 1997 '' ( 1997 ) `` Written in the Stars '' ( 1999 ) 2000s `` I Want
Love '' ( 2001 ) `` This Train Do n't Stop There Anymore '' ( 2002 ) `` Original Sin '' ( 2002 ) `` Are You Ready for Love '' ( 2003 ) `` Ghetto Gospel '' (
2005 ) `` Electricity '' ( 2005 ) `` The Bridge '' ( 2006 ) `` Joseph , Better You than Me '' ( 2008 ) `` Tiny Dancer ( Hold Me Closer ) '' ( 2009 ) Related
articles Albums discography Videography Book : Elton John Category : Elton John Ivor Novello Awards -- Best Song Musically and Lyrically 1960s She 's Leaving
Home ( 1968 ) 1970s Where Do You Go To ( My Lovely ) ? ( 1970 ) Something ( 1971 ) Do n't Let It Die ( 1972 ) Without You ( 1973 ) Daniel ( 1974 ) Streets
</item 10>
=== END ITEM id=8dcd20bf56209cfd ===

=== ITEM id=73e0de527f75e3f2 ===
Gold sentences and facts:
Sentence 0: The uvea is the vascular middle layer of the eye . It is traditionally divided into three areas , from front to back , the :
  Fact 0.0: The uvea is the vascular middle layer of the eye.
  Fact 0.1: The uvea is traditionally divided into three areas.
  Fact 0.2: The three areas of the uvea are divided from front to back.

Retrieved items:
<item 1>
Retina
Inner nuclear layer -- contains the nuclei and surrounding cell bodies ( perikarya ) of the amacrine cells , bipolar cells , and horizontal cells . Outer
plexiform layer -- projections of rods and cones ending in the rod spherule and cone pedicle , respectively . These make synapses with dendrites of bipolar
cells . In the macular region , this is known as the Fiber layer of Henle . Outer nuclear layer -- cell bodies of rods and cones . External limiting membrane --
layer that separates the inner segment portions of the photoreceptors from their cell nucleus . Layer of rods and cones -- layer of rod cells and cone cells .
Retinal pigment epithelium -- single layer of cuboidal cells ( with extrusions not shown in diagram ) . This is closest to the choroid . These layers can be
grouped into 4 main processing stages : photoreception ; transmission to bipolar cells ; transmission to ganglion cells , which also contain photoreceptors ,
the photosensitive ganglion cells ; and transmission along the optic nerve . At each synaptic stage there are also laterally connecting horizontal and amacrine
cells .
</item 1>

<item 2>
Retina
In section , the retina is no more than 0.5 mm thick . It has three layers of nerve cells and two of synapses , including the unique ribbon synapse . The optic
nerve carries the ganglion cell axons to the brain , and the blood vessels that supply the retina . The ganglion cells lie innermost in the eye while the
photoreceptive cells lie beyond . Because of this counter-intuitive arrangement , light must first pass through and around the ganglion cells and through the
thickness of the retina , ( including its capillary vessels , not shown ) before reaching the rods and cones . Light is absorbed by the retinal pigment
epithelium or the choroid ( both of which are opaque ) . The white blood cells in the capillaries in front of the photoreceptors can be perceived as tiny bright
moving dots when looking into blue light . This is known as the blue field entoptic phenomenon ( or Scheerer 's phenomenon ) . Between the ganglion cell layer
and the rods and cones there are two layers of neuropils where synaptic contacts are made . The neuropil layers are the outer plexiform layer and the inner
plexiform layer .
</item 2>

<item 3>
Uvea
uva , grape ) , also called the uveal layer , uveal coat , uveal tract , or vascular tunic , is the pigmented middle of the three concentric layers that make up
an eye . The name is possibly a reference to its reddish - blue or almost black colour , wrinkled appearance and grape - like size and shape when stripped
intact from a cadaveric eye . Its use as a technical term in anatomy and ophthalmology is relatively modern . Contents ( hide ) 1 Structure 1.1 Regions 2
Function 3 Pharmacology 4 Immunology 5 Clinical significance 6 References 7 External links Structure ( edit ) Regions ( edit ) The uvea is the vascular middle
layer of the eye . It is traditionally divided into three areas , from front to back , the : Iris Ciliary body Choroid Function ( edit ) The prime functions of
the uveal tract as a unit are : Nutrition and gas exchange : uveal vessels directly perfuse the ciliary body and iris , to support their metabolic needs , and
indirectly supply diffusible nutrients to the outer retina , sclera , and lens , which lack any intrinsic blood supply .
</item 3>

<item 4>
Uvea
M cell , K cell , Muller glia Other Macula Perifoveal area Parafoveal area Fovea Foveal avascular zone Foveola Optic disc Optic cup Ora serrata Anatomical
regions of the eye Anterior segment Adnexa Fibrous tunic Anterior chamber Aqueous humour Iris Posterior chamber Ciliary body Lens Capsule of lens Zonule of Zinn
Posterior segment Vitreous chamber Vitreous body Retina Choroid Other Keratocytes Ocular immune system Optical coherence tomography Eye care professional Eye
disease Refractive error Accommodation Physiological Optics Visual perception Retrieved from `` https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Retina
The layers and anatomical correlation are as follows : OCT scan of the macular area of a retina at 800 nm , axial resolution 3 μm From innermost to outermost ,
the layers identifiable by OCT are as follows : # OCT Layer / Conventional Label Anatomical Correlate Reflectivity on OCT Specific anatomical boundaries ?
Additional references Posterior cortical vitreous Posterior cortical vitreous Hyper - reflective Yes Preretinal space In eyes where the vitreous has fully or
partially detached from the retina , this is the space created between the posterior cortical vitreous face and the internal limiting membrane of the retina .
Hypo - reflective Internal limiting membrane ( ILM ) Formed by Müller cell endfeet ( unclear if it can be observed on OCT ) Hyper - reflective No Nerve fiber
layer ( NFL ) Ganglion cell axons travelling towards the optic nerve Ganglion cell layer ( GCL ) Ganglion cell bodies ( and some displaced amacrine cells ) Hypo
- reflective 5 Inner plexiform layer ( IPL ) Synapses between bipolar , amacrine and ganglion cells Hyper - reflective 6 Inner nuclear layer ( INL ) a )
Horizontal ,
</item 5>

<item 6>
Retina
The optics of the eye create a focused two - dimentional image of the visual world on the retina , which translates that image into electrical neural impulses
to the brain to create visual perception , the retina serving much the same function as film or a CCD in a camera . The retina consists of several layers of
neurons interconnected by synapses . The neural retina refers to the three layers of neural cells ( photo receptor cells , bipolar cells , and ganglion cells )
within the retina , which in its entirety comprises ten distinct layers , including an outer layer of pigmented epithelial cells . The only neural cells that
are directly sensitive to light are the photoreceptor cells , which are of two types : rods and cones . Rods function mainly in dim light and provide black -
and - white vision while cones are responsible for the perception of colour . A third type of photoreceptor , the photosensitive ganglion cells , is important
for entrainment and reflexive responses to the brightness of light . Light striking the retina initiates a cascade of chemical and electrical events that
ultimately trigger nerve impulses that are sent to various visual centres of the brain through the fibres of the optic nerve .
</item 6>

<item 7>
Retina
The optic disc is at right , and the macula lutea is near the centre . The retina is stratified into distinct layers , each containing specific cell types or
cellular compartments that have metabolisms with different nutritional requirements . To satisfy these requirements , the ophthalmic artery bifurcates and
supplies the retina via two distinct vascular networks : the choroidal network , which supplies the choroid and the outer retina , and the retinal network ,
which supplies the retina 's inner layer . Circulatory mechanisms ( edit ) At first glance , one may think that the vertebrate retina is `` wired wrongly '' or
`` badly designed '' ; but in fact , the retina could not function if it were not inverted . The photoreceptor layer must be embedded in the retinal pigment
epithelium ( RPE ) , which performs at least seven vital functions , one of the most obvious being to supply oxygen and other necessary nutrients needed for the
photoreceptors to function . These nutrients include glucose , fatty acids , and retinal .
</item 7>

<item 8>
Uvea
Uvea - wikipedia Uvea Jump to : navigation , search For the Pacific island and chiefdom in Wallis and Futuna , see Wallis ( island ) and Uvea ( Wallis and
Futuna ) . For the Pacific commune and island in New Caledonia , see Ouvéa and Ouvéa Island . This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( February 2014 ) ( Learn how and when to
remove this template message ) Uvea Horizontal section of the eyeball . The constituents of the uvea follow : iris labeled at top , ciliary body labeled at
upper right , and choroid labeled at center right . ) Details Identifiers Latin tunica vasculosa bulbi MeSH D014602 TA A15. 2.03. 001 FMA 58103 Anatomical
terminology ( edit on Wikidata ) The uvea ( / ˈjuːvɪə / ) ( Lat .
</item 8>

<item 9>
Uvea
( The cornea has no adjacent blood vessels and is oxygenated by direct gas exchange with the environment . ) Light absorption : the uvea improves the contrast
of the retinal image by reducing reflected light within the eye ( analogous to the black paint inside a camera ) , and also absorbs outside light transmitted
through the sclera , which is not fully opaque . In addition some uveal regions have special functions of great importance , including secretion of the aqueous
humour by the ciliary processes , control of accommodation ( focus ) by the ciliary body , and optimisation of retinal illumination by the iris 's control over
the pupil . Many of these functions are under the control of the autonomic nervous system . Pharmacology ( edit ) The pupil provides a visible example of the
neural feedback control in the body . This is subserved by a balance between the antagonistic sympathetic and parasympathetic divisions of the autonomic nervous
system . Informal pharmacological experiments have been performed on the pupil for centuries , since the pupil is readily visible , and its size can be readily
altered by applying drugs -- even crude plant extracts -- to the cornea .
</item 9>

<item 10>
Uvea
com ( hide ) Anatomy of the globe of the human eye Fibrous tunic ( outer ) Sclera Episcleral layer Schlemm 's canal Trabecular meshwork Cornea Limbus layers
Epithelium Bowman 's Stroma Dua 's layer Descemet 's Endothelium Uvea / vascular tunic ( middle ) Choroid Capillary lamina of choroid Bruch 's membrane Sattler
's layer Ciliary body Ciliary processes Ciliary muscle Pars plicata Pars plana Iris Stroma Pupil Iris dilator muscle Iris sphincter muscle Retina ( inner )
Layers Inner limiting membrane Nerve fiber layer Ganglion cell layer Inner plexiform layer Inner nuclear layer Outer plexiform layer Outer nuclear layer
External limiting membrane Layer of rods and cones Retinal pigment epithelium Cells Photoreceptor cells ( Cone cell , Rod cell ) → ( Horizontal cell ) → Bipolar
cell → ( Amacrine cell ) → Retina ganglion cell ( Midget cell , Parasol cell , Bistratified cell , Giant retina ganglion cells , Photosensitive ganglion cell )
→ Diencephalon : P cell , M cell , K cell ,
</item 10>
=== END ITEM id=73e0de527f75e3f2 ===

=== ITEM id=24ee271898a1d6cd ===
Gold sentences and facts:
Sentence 0: Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ; sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
  Fact 0.0: Lynne is the eldest of the Slater sisters.
  Fact 0.1: Lynne arrives in Albert Square in 2000.
  Fact 0.2: Lynne arrives in Albert Square with her family.
  Fact 0.3: Lynne's father is Charlie, played by Derek Martin.
  Fact 0.4: Lynne's sister Kat is played by Jessie Wallace.
  Fact 0.5: Lynne's sister Little Mo is played by Kacey Ainsworth.
  Fact 0.6: Lynne's sister Zoe is played by Michelle Ryan.
  Fact 0.7: Lynne's grandmother is Mo, played by Laila Morse.
  Fact 0.8: Unlike Little Mo and Zoe, Lynne is aware that Zoe is in fact Kat's daughter.
  Fact 0.9: Lynne helps keep the family secret that Zoe is Kat's daughter until Kat reveals the truth to Zoe.

Retrieved items:
<item 1>
Lynne Hobbs
Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie McFarlane Poppy Meadow Darren Miller Demi Miller Keith Miller Mickey Miller Rosie Miller Rob Minter
Archie Mitchell Ben Mitchell Danny Mitchell Glenda Mitchell Grant Mitchell Jamie Mitchell Kate Mitchell Little Mo Mitchell Peggy Mitchell Ronnie Mitchell Roxy
Mitchell Sam Mitchell Tiffany Mitchell Vinnie Monks Alfie Moon Anthony Moon Danny Moon Eddie Moon Jake Moon Michael Moon Nana Moon Spencer Moon Tyler Moon
Trevor Morgan Rod Norman Andy O'Brien Mercy Olubunmi Ali Osman Mehmet Osman Sue Osman Steve Owen Annie Palmer George Palmer Lola Pearce Minty Peterson Frankie
Pierre Paul Priestly Irene Raymond Louise Raymond Simon Raymond Terry Raymond Dennis Rickman Binnie Roberts Roly Willy Roper Matthew Rose Colin Russell Mandy
Salter Trevor Short Lydia Simmonds Ethel Skinner Charlie Slater Sean Slater Zoe Slater Leon Small Babe Smith Gus Smith Mary Smith Dan Sullivan Gavin Sullivan
Dawn Swann Celestine and Etta Tavernier Clyde Tavernier Hattie Tavernier Jules Tavernier Kelly Taylor Heather Trott Anthony Trueman Paul Trueman Yolande Trueman
Owen Turner Angie Watts Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe
</item 1>

<item 2>
Lynne Hobbs
Lynne is a fierce judge of right and wrong , although she is not always able to practise what she preaches . She has a turbulent relationship with husband Garry
( Ricky Groves ) , but despite her complaints about him , it ends up being Lynne 's adultery that puts their relationship in jeopardy long before Garry strays .
Contents ( hide ) 1 Storylines 1.1 Backstory 1.2 2000 -- 2004 2 Creation and development 3 In popular culture 4 References 5 External links Storylines ( edit )
Backstory ( edit ) Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ;
sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne
is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
</item 2>

<item 3>
Lynne Hobbs
Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe Wicks Kevin Wicks Lorraine Wicks Simon Wicks Debbie Wilkins Cindy Williams
James Willmott - Brown Willy May Wright Lists of characters By year : 1985 1986 2017 2018 Others : Flaherty family Dickens Hill inmates The Banned band members
Characters from spin - offs Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lynne_Hobbs&oldid=838104750 '' Categories : EastEnders characters
Fictional characters introduced in 2000 Female characters in television Hidden categories : Articles needing additional references from March 2011 All articles
needing additional references All articles with unsourced statements Articles with unsourced statements from January 2011 Use dmy dates from March 2011 Talk
Contents About Wikipedia Gaeilge Edit links This page was last edited on 24 April 2018 , at 23 : 33 . About Wikipedia
</item 3>

<item 4>
Sheila Carter
and Colleen Daniel and Lily Devon and Hilary `` Nadia 's Theme '' ( hide ) The Young and the Restless characters Present and future characters Ashley Abbott
Billy Abbott Jack Abbott Jill Abbott John Abbott Johnny Abbott Traci Abbott Cane Ashby Charlie Ashby Mattie Ashby Sam Ashby Lily Winters Colin Atkinson Fenmore
Baldwin Michael Baldwin Gloria Abbott Bardwell Jeffrey Bardwell Christine Blair Mariah Copeland Hilary Curtis Lauren Fenmore Kevin Fisher Scotty Grainger Devon
Hamilton Reed Hellstrom Anita Lawson Dina Mergeron Chloe Mitchell Leslie Michaelson Abby Newman Chelsea Lawson Christian Newman Connor Newman Faith Newman Katie
Newman Nicholas Newman Nikki Newman Noah Newman Sharon Newman Summer Newman Victor Newman Victoria Newman Stitch Rayburn Barton Shelby Phyllis Summers Esther
Valentine Paul Williams Neil Winters ( hide ) Past characters Kyle Abbott Kelly Andrews Genevieve Atkinson Avery Bailey Clark Eden Baldwin Isabella Braña
Mackenzie Browning Brad Carlton Colleen Carlton Daisy Carter Sheila Carter Chance Chancellor Katherine Chancellor Joe Clark Matt Clark Greg Foster Dr. Snapper
Foster Sam Gibson Diego Guittierez Nathan Hastings Jana Hawkes J.T.
</item 4>

<item 5>
List of The Fall characters
Katie is a gifted musician and uses online videos to express her feelings . During series 2 she becomes deeply embroiled in Paul 's life . Paul ties her up in a
hotel room , for example , whilst Katie breaks into a house wearing a shirt with Spector 's face on . Benedetto is placed under police surveillance and later
arrested whilst trying to dispose of evidence that implicates Paul in the murders . In series three , she attacks a girl with acid , and is detained for her
crimes . During this time , she reveals to Stella that her father 's death had a severe emotional impact on her , and she has been self - harming . Sally Ann
Spector ( edit ) Sally Ann Spector ( Bronagh Waugh ) , the wife of Paul Spector and a neonatal nurse . She has a deep devotion to Paul , but is wary of his
character as she has never met his family . Sally Ann is the mother of his two children and is unaware of his dark side . Over the course of the series , she
becomes embroiled in his crimes without her knowledge , and even provides a false alibi to the police .
</item 5>

<item 6>
Balamory
Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name (
Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1
- 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the
Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little
Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet ,
</item 6>

<item 7>
Lynne Hobbs
Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986 ) Dotty Cotton Nick Cotton Stella Crawford Beppe di Marco
Bruno and Luisa di Marco Gianni di Marco Rosa di Marco Ray Dixon Jase Dyer Huw Edwards Steve Elliot Nellie Ellis Barry Evans Natalie Evans Roy Evans Fatboy
Ferreira family Conor Flaherty Mary Flaherty Fred Fonseca Arthur Fowler Lisa Fowler Mark Fowler Michelle Fowler Pauline Fowler Ruth Fowler Vicki Fowler Chelsea
Fox Libby Fox Joanne Francis Jodie Gold Vanessa Gold Marge Green Nina Harris Dexter Hartman Julie Haye Alex Healy Christine Hewitt Ted Hills Tony Hills Garry
Hobbs Lynne Hobbs Lofty Holloway Vincent Hubbard Andy Hunter Alan Jackson Bianca Jackson Billie Jackson Blossom Jackson Carmel Jackson Carol Jackson Naima
Jeffery Saeed Jeffery Al Jenkins Greg Jessop Lucas Johnson Danielle Jones Naomi Julien Gita Kapoor Sanjay Kapoor Karim family Yusef Khan Tony King Harold Legg
Donna Ludlow Joe Macer Ryan Malloy Jill Marsden Afia Masood Amira Masood Shabnam Masood Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie
</item 7>

<item 8>
Lynne Hobbs
In January 2016 , Lynne attends Charlie 's funeral off - screen . Creation and development ( edit ) In 2000 , EastEnders ' executive producer John Yorke decided
to introduce the `` classic '' Slater family . He felt the show needed to go back to its roots and bring back some traditional values . BBC 's head of drama ,
Mal Young commented , `` We do not have enough solid families in the soap , there were a lot of fractured families and people who were alone . '' The family
were created as a replacement for the di Marco family , who were axed by Yorke . The family consisted of , grandmother Mo Harris ( Laila Morse ) , father
Charlie ( Derek Martin ) , his four children , Lynne ( Elaine Lordan ) , Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) , Zoe ( Michelle Ryan ) , as
well as Lynne 's boyfriend Garry Hobbs ( Ricky Groves ) . It later transpired that Kat was the mother of Zoe , after being raped by her uncle Harry Slater (
Michael Elphick ) .
</item 8>

<item 9>
Lynne Hobbs
Unsourced material may be challenged and removed . ( March 2011 ) ( Learn how and when to remove this template message ) Lynne Hobbs EastEnders character
Portrayed by Elaine Lordan Duration 2000 -- 2004 First appearance Episode 2003 18 September 2000 ( 2000 - 09 - 18 ) Last appearance Episode 2756 2 July 2004 (
2004 - 07 - 02 ) Created by Tony Jordan Introduced by John Yorke Spin - off appearances Slaters in Detention ( 2003 ) Classification Former ; regular Profile
Other names Lynne Slater Occupation Café worker ( show ) Family Father Charlie Slater Mother Viv Slater Sisters Kat Slater Little Mo Mitchell Belinda Peacock
Husband Garry Hobbs ( 2001 -- 2004 ) Daughters Vivienne Hobbs Grandmothers Mo Harris Uncles Harry Slater Nephews Freddie Mitchell Nieces Zoe Slater Lynne Hobbs
( also Slater ) is a fictional character from the BBC soap opera EastEnders , played by Elaine Lordan from September 2000 to July 2004 .
</item 9>

<item 10>
Lynne Hobbs
links ( edit ) Lynne Hobbs at BBC Online EastEnders characters Present and future characters Kathy Beale Sharon Watts Ian Beale Dot Cotton Martin Fowler Phil
Mitchell Sonia Fowler Robbie Jackson Mel Owen Billy Mitchell Mo Harris Kat Slater Patrick Trueman Louise Mitchell Bobby Beale Stacey Slater Jean Slater Honey
Mitchell Denise Fox Max Branning Janet Mitchell Shirley Carter Jay Brown Rainie Cross Masood Ahmed Jack Branning Whitney Dean Tiffany Butcher Kim Fox Cora Cross
Tina Carter Linda Carter Mick Carter Charlie Cotton ( hide ) Past characters AJ Ahmed Della Alexander Johnny Allen Ruby Allen Aunt Sal Tom Banks Geoff Barnes
Clare Bates Debbie Bates Laurie Bates Nigel Bates Cindy Beale Jane Beale Laura Beale Lou Beale Lucy Beale Pete Beale Peter Beale Steven Beale Polly Becker Adam
Best Manda Best Abi Branning Alice Branning April Branning Bradley Branning Derek Branning Jim Branning Lauren Branning Suzy Branning Tanya Branning Aidan
Brosnan Diane Butcher Frank Butcher Janine Butcher Liam Butcher Mo Butcher Pat Butcher Ricky Butcher Kelvin Carpenter Tony Carpenter Nancy Carter Stan Carter
Zsa Zsa Carter Lorna Cartwright Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986
</item 10>
=== END ITEM id=24ee271898a1d6cd ===

=== ITEM id=4ec5243c12c91e73 ===
Gold sentences and facts:
Sentence 0: Sanitary napkins ( also called sanitary towels or pads ) -- Rectangular pieces of material worn attached to the underwear to absorb menstrual flow , often with an adhesive backing to hold the pad in place . Disposable pads may contain wood pulp or gel products , usually with a plastic lining and bleached . Tampons -- Disposable cylinders of treated rayon / cotton blends or all - cotton fleece , usually bleached , that are inserted into the vagina to absorb menstrual flow . Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped device worn inside the vagina to collect menstrual flow .
  Fact 0.0: Sanitary napkins are also called sanitary towels or pads.
  Fact 0.1: Sanitary napkins are rectangular pieces of material worn attached to the underwear to absorb menstrual flow.
  Fact 0.2: Sanitary napkins are often given an adhesive backing to hold the pad in place.
  Fact 0.3: Disposable sanitary pads may contain wood pulp or gel products.
  Fact 0.4: Disposable sanitary pads usually have a plastic lining and are bleached.
  Fact 0.5: Tampons are disposable cylinders of treated rayon / cotton blends or all-cotton fleece.
  Fact 0.6: Tampons are usually bleached.
  Fact 0.7: Tampons are inserted into the vagina to absorb menstrual flow.
  Fact 0.8: Disposable menstrual cups are made of soft plastic.
  Fact 0.9: Disposable menstrual cups are a firm, flexible, cup-shaped device.
  Fact 0.10: Disposable menstrual cups are worn inside the vagina.
  Fact 0.11: Disposable menstrual cups collect menstrual flow.
Sentence 1: Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped device worn inside the vagina to collect menstrual flow .
  Fact 1.0: Disposable menstrual cups are made of soft plastic.
  Fact 1.1: Disposable menstrual cups are a firm, flexible, cup-shaped device.
  Fact 1.2: Disposable menstrual cups are worn inside the vagina.
  Fact 1.3: Disposable menstrual cups collect menstrual flow.
Sentence 2: Menstrual cups -- A firm , flexible bell - shaped device worn inside the vagina to collect menstrual flow . Menstrual cups are usually made of silicone and can last 5 years or longer .
  Fact 2.0: Menstrual cups are a firm, flexible, bell-shaped device.
  Fact 2.1: Menstrual cups are worn inside the vagina.
  Fact 2.2: Menstrual cups collect menstrual flow.
  Fact 2.3: Menstrual cups are usually made of silicone.
  Fact 2.4: Menstrual cups can last 5 years or longer.

Retrieved items:
<item 1>
Menstruation
Menstrual cups are usually made of silicone and can last 5 years or longer . Reusable cloth pads -- Pads that are made of cotton ( often organic ) , terrycloth
, or flannel , and may be handsewn ( from material or reused old clothes and towels ) or storebought . Padded panties or period - proof underwear -- Reusable
cloth ( usually cotton ) underwear with extra absorbent layers sewn in to absorb flow . Some also use patented technology to be leak resistant , such as the
brand THINX . Sea sponges -- Natural sponges , worn internally like a tampon to absorb menstrual flow . Blanket , towel -- ( also known as a draw sheet ) --
large reusable piece of cloth , most often used at night , placed between legs to absorb menstrual flow . Non-commercial items In developing countries , many
poor women can not afford commercial feminine hygiene products . Instead , they use materials found in the environment or other improvised materials .
</item 1>

<item 2>
Graph query row:
item: Menstrual cup
kind: reusable device
relationship: PART_OF
related: Reusable products
related_kind: product category
fact: None
context: None
relationship_detail: None
relation_detail: None
role: None
</item 2>

<item 3>
Graph query row:
item: Menstrual cup
kind: reusable device
relationship: RELATED_TO
related: Vagina
related_kind: None
fact: None
context: None
relationship_detail: None
relation_detail: None
role: None
</item 3>

<item 4>
Menstruation
Some anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle . However , anthropologist Beverly Strassmann has
proposed that the energy savings of not having to continuously maintain the uterine lining more than offsets energy cost of having to rebuild the lining in the
next fertility cycle , even in species such as humans where much of the lining is lost through bleeding ( overt menstruation ) rather than reabsorbed ( covert
menstruation ) . Many have questioned the evolution of overt menstruation in humans and related species , speculating on what advantage there could be to losing
blood associated with dismantling the endometrium , rather than absorbing it , as most mammals do . Humans do , in fact , reabsorb about two - thirds of the
endometrium each cycle . Strassmann asserts that overt menstruation occurs not because it is beneficial in itself . Rather , the fetal development of these
species requires a more developed endometrium , one which is too thick to reabsorb completely . Strassman correlates species that have overt menstruation to
those that have a large uterus relative to the adult female body size .
</item 4>

<item 5>
Menstruation
The intention behind this decision was the hope of the inventor , John Rock , to win approval for his invention from the Roman Catholic Church . That attempt
failed , but the 28 - day cycle remained the standard when the pill became available to the public . There is debate among medical researchers about the
potential long - term impacts of these practices upon female health . Some researchers point to the fact that historically , females have had far fewer
menstrual periods throughout their lifetimes , a result of shorter life expectancies , as well as a greater length of time spent pregnant or breast - feeding ,
which reduced the number of periods experienced by females . These researchers believe that the higher number of menstrual periods by females in modern
societies may have a negative impact upon their health . On the other hand , some researchers believe there is a greater potential for negative impacts from
exposing females perhaps unnecessarily to regular low doses of synthetic hormones over their reproductive years . Breastfeeding Main article : Lactational
amenorrhea method Breastfeeding causes negative feedback to occur on pulse secretion of gonadotropin - releasing hormone ( GnRH ) and luteinizing hormone ( LH )
.
</item 5>

<item 6>
Menstruation
Secular culture Anthropologists , Lock and Nguyen ( 2010 ) , have noted that the heavy medicalization of the reproductive life - stages of women in the West ,
mimic power structures that are deemed , in other cultural practices , to function as a form of `` social control '' . Medicalization of the stages of women 's
lives , such as birth and menstruation , has enlivened a feminist perspective that investigates the social implications of biomedicine 's practice . `` ( C )
ultural analysis of reproduction ... attempts to show how women ... exhibit resistance and create dominant alternative meanings about the body and reproduction
to those dominant among the medical profession . '' Other In some parts of South Asia , women are isolated during menstruation . In 2005 , in Nepal , the
Supreme Court abolished the practice of chhaupadi , keeping women in cow - sheds during menstruation . Sexual activity Sexual intercourse during menstruation
does not cause damage in and of itself , but the woman 's body is more vulnerable during this time .
</item 6>

<item 7>
Graph query row:
item: Menstrual cup
kind: reusable device
relationship: RELATED_TO
related: Menstrual flow
related_kind: flow
fact: None
context: None
relationship_detail: None
relation_detail: None
role: None
</item 7>

<item 8>
Menstruation
Instead , they use materials found in the environment or other improvised materials . Absorption materials that may be used by women who can not afford anything
else include : sand , ash , small hole in earth , cloth , whole leaf , leaf fibre ( such as water hyacinth , banana , papyrus , cotton fibre ) , paper ( toilet
paper , re-used newspaper , pulped and dried paper ) , animal pelt e.g. goat skin , double layer of underwear , skirt or sari . Society and culture Amra Padatik
India , celebration of Menstrual Hygiene Day in India Traditions and taboos Main article : Culture and menstruation Further information : Menstrual taboo Many
religions have menstruation - related traditions . Islam Islam prohibits sexual contact with women during menstruation in the 2nd chapter of the Quran . Women
are exempted from Salah ( prayer ) . They are also prohibited from fasting and obligatory fasts of Ramadan are to be performed during other days . During menses
pilgrimages are allowed ; however , circumambulation of the Kaaba is prohibited and is to be performed during other times .
</item 8>

<item 9>
Menstruation
Menstruation : A Cultural History . Palgrave Macmillan . ISBN 1 - 4039 - 3935 - 7 . Retrieved 9 November 2013 . Knight , Chris ( 1995 ) . Blood Relations :
Menstruation and the Origins of Culture . New Haven and London : Yale University . ISBN 0 - 300 - 04911 - 0 . Retrieved 9 November 2013 . External links
Wikiquote has quotations related to : Menstruation Wikimedia Commons has media related to Menstruation . Museum of Menstruation Human physiology of sexual
reproduction Menstrual cycle Menarche Menstruation Follicular phase Ovulation Luteal phase Gametogenesis Spermatogenesis ( spermatogonium spermatocyte spermatid
sperm ) Oogenesis ( oogonium oocyte ootid ovum ) Germ cell ( gonocyte gamete ) Human sexual behavior Sexual arousal Sexual intercourse Masturbation Erection
Orgasm Ejaculation Insemination Fertilisation / Fertility Implantation Pregnancy Postpartum period Mechanics of sex Life span Prenatal development / Sexual
dimorphism / Sexual differentiation ( Feminization Virilization
</item 9>

<item 10>
Menstruation
Menstrual products ( also called `` feminine hygiene '' products ) are made to absorb or catch menstrual blood . A number of different products are available -
some are disposable , some are reusable . Where women can afford it , items used to absorb or catch menses are usually commercially manufactured products .
Disposable products Disposable sanitary napkin Tampon in plastic applicator Sanitary napkins ( also called sanitary towels or pads ) -- Rectangular pieces of
material worn attached to the underwear to absorb menstrual flow , often with an adhesive backing to hold the pad in place . Disposable pads may contain wood
pulp or gel products , usually with a plastic lining and bleached . Tampons -- Disposable cylinders of treated rayon / cotton blends or all - cotton fleece ,
usually bleached , that are inserted into the vagina to absorb menstrual flow . Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped
device worn inside the vagina to collect menstrual flow . Reusable products Menstrual cup Cloth menstrual pad Menstrual cups -- A firm , flexible bell - shaped
device worn inside the vagina to collect menstrual flow . Menstrual cups are usually made of silicone and can last 5 years or longer .
</item 10>
=== END ITEM id=4ec5243c12c91e73 ===

=== ITEM id=1370321ec5dc494e ===
Gold sentences and facts:
Sentence 0: Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio .
  Fact 0.0: Much of the humus in most soils has persisted for more than 100 years rather than having been decomposed into CO.
  Fact 0.1: The long-persisting humus in most soils can be regarded as stable.
  Fact 0.2: Much of the humus in most soils has been protected from decomposition by microbial or enzyme action.
  Fact 0.3: Protected humus is hidden (occluded) inside small aggregates of soil particles, or is tightly sorbed or complexed to clays.
  Fact 0.4: Most humus that is not protected in this way is decomposed within 10 years.
  Fact 0.5: Humus that is not protected in this way can be regarded as less stable or more labile.
  Fact 0.6: Stable humus contributes few plant-available nutrients in soil.
  Fact 0.7: Stable humus helps maintain the physical structure of soil.
  Fact 0.8: A very stable form of humus is formed from the slow oxidation of soil carbon after finely powdered charcoal is incorporated into the topsoil.
  Fact 0.9: The formation of the very stable humus from charcoal is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio.

Retrieved items:
<item 1>
Humus
Stability ( edit ) Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as
stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of
soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as
less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable
form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated
to have been important in the formation of the very fertile Amazonian terra preta do Indio . Horizons ( edit ) Humus has a characteristic black or dark brown
color and is organic due to an accumulation of organic carbon . Soil scientists use the capital letters O , A , B , C , and E to identify the master horizons ,
and lowercase letters for distinctions of these horizons .
</item 1>

<item 2>
Humus
Most soils have three major horizons : the surface horizon ( A ) , the subsoil ( B ) , and the substratum ( C ) . Some soils have an organic horizon ( O ) on
the surface , but this horizon can also be buried . The master horizon ( E ) is used for subsurface horizons that have significantly lost minerals ( eluviation
) . Bedrock , which is not soil , uses the letter R . Benefits of soil organic matter and humus ( edit ) The importance of chemically stable humus is thought by
some to be the fertility it provides to soils in both a physical and chemical sense , though some agricultural experts put a greater focus on other features of
it , such as its ability to suppress disease . It helps the soil retain moisture by increasing microporosity , and encourages the formation of good soil
structure . The incorporation of oxygen into large organic molecular assemblages generates many active , negatively charged sites that bind to positively
charged ions ( cations ) of plant nutrients , making them more available to the plant by way of ion exchange . Humus allows soil organisms to feed and reproduce
, and is often described as the `` life - force '' of the soil .
</item 2>

<item 3>
Humus
Soil Biology and Biochemistry . 26 ( 2 ) : 183 -- 220 . doi : 10.1016 / 0038 - 0717 ( 94 ) 90161 - 9 . Jump up ^ Humintech ® Definition Of Soil Organic Matter &
Humic Acids Based Products Jump up ^ Brady , N.C. ; Weil , R.R. ( 1999 ) . The nature and properties of soils . Upper Saddle River , N.J. : Prentice Hall , Inc
. Jump up ^ Soil biology Jump up ^ Berg , B. , McClaugherty , C. , 2007 . Plant litter : decomposition , humus formation , carbon sequestration , 2nd ed .
Springer , 338 pp. , ISBN 3 - 540 - 74922 - 5 Jump up ^ Levin , L. ; Forchiassin , F. ; Ramos , A.M. ( 2002 ) . `` Copper induction of lignin - modifying
enzymes in the white - rot fungus Trametes trogii '' . Mycologia. 94 : 377 -- 383 .
</item 3>

<item 4>
Humus
Humus has many nutrients that improve the health of soil , nitrogen being the most important . The ratio of carbon to nitrogen ( C : N ) of humus is 10 : 1 .
Contents ( hide ) 1 Description 2 Humification 3 Stability 4 Horizons 5 Benefits of soil organic matter and humus 6 See also 7 References 8 External links
Description ( edit ) It is difficult to define humus precisely because it is a very complex substance which is not fully understood . Humus is different from
decomposing soil organic matter . The latter looks rough and has visible remains of the original plant or animal matter . Fully humified humus , on the contrary
, has a uniformly dark , spongy , and jelly - like appearance , and is amorphous ; it may gradually decompose over several years or persist for millennia . It
has no determinate shape , structure , or quality . However , when examined under a microscope , humus may reveal tiny plant , animal , or microbial remains
that have been mechanically , but not chemically , degraded . This suggests an ambiguous boundary between humus and soil organic matter .
</item 4>

<item 5>
Humus
, 2011 . Accessed 24 Nov 2011 . < http://www.britannica.com/EBchecked/topic/276408/humus > . Jump up ^ Chertov , O.G. ; Kornarov , A.S. ; Crocker , G. ; Grace ,
P. ; Klir , J. ; Körschens , M. ; Poulton , P.R. ; Richter , D. ( 1997 ) . `` Simulating trends of soil organic carbon in seven long - term experiments using
the SOMM model of the humus types '' . Geoderma. 81 : 121 -- 135 . doi : 10.1016 / S0016 - 7061 ( 97 ) 00085 - 2 . Jump up ^ Baritz , R. , 2003 . Humus forms in
forests of the northern German lowlands . Schweizerbart , Stuttgart , Germany , 145 pp. ( 2 ) Jump up ^ Bunting , B.T. ; Lundberg , J. ( 1995 ) .
</item 5>

<item 6>
Humus
Humus is a colloidal substance and increases the cation exchange capacity of soil , hence its ability to store nutrients by chelation . While these nutrient
cations are available to plants , they are held in the soil and prevented from being leached by rain or irrigation . Humus can hold the equivalent of 80 -- 90 %
of its weight in moisture , and therefore increases the soil 's capacity to withstand drought . The biochemical structure of humus enables it to moderate , i.e.
buffer , excessive acidic or alkaline soil conditions . During humification , microbes secrete sticky , gum - like mucilages ; these contribute to the crumby
structure ( tilth ) of the soil by adhering particles together and allowing greater aeration of the soil . Toxic substances such as heavy metals and excess
nutrients can be chelated , i.e. , bound to the organic molecules of humus , and so prevented from leaching away . The dark , usually brown or black , color of
humus helps to warm cold soils in Spring .
</item 6>

<item 7>
Humus
Plant remains , including those that animals digested and excreted , contain organic compounds : sugars , starches , proteins , carbohydrates , lignins , waxes
, resins , and organic acids . Decay in the soil begins with the decomposition of sugars and starches from carbohydrates , which decompose easily as
detritivores initially invade the dead plant organs , while the remaining cellulose and lignin decompose more slowly . Simple proteins , organic acids ,
starches , and sugars decompose rapidly , while crude proteins , fats , waxes , and resins remain relatively unchanged for longer periods of time . Lignin ,
which is quickly transformed by white - rot fungi , is one of the primary precursors of humus , together with by - products of microbial and animal activity .
The humus produced by humification is thus a mixture of compounds and complex biological chemicals of plant , animal , or microbial origin that has many
functions and benefits in soil . Some judge earthworm humus ( vermicompost ) to be the optimal organic manure .
</item 7>

<item 8>
Humus
Humus has a characteristic black or dark brown color and is an accumulation of organic carbon . Besides the three major soil horizons of ( A ) surface / topsoil
, ( B ) subsoil , and ( C ) substratum , some soils have an organic horizon ( O ) on the very surface . Hard bedrock ( R ) is not in a strict sense soil . In
soil science , humus ( derived in 1790 -- 1800 from the Latin humus for earth , ground ) denominates the fraction of soil organic matter that is amorphous and
without the `` cellular cake structure characteristic of plants , micro-organisms or animals . '' Humus significantly affects the bulk density of soil and
contributes to its retention of moisture and nutrients . In agriculture , `` humus '' sometimes also is used to describe mature or natural compost extracted
from a woodland or other spontaneous source for use as a soil conditioner . It is also used to describe a topsoil horizon that contains organic matter ( humus
type , humus form , humus profile ) . Humus is the dark organic matter that forms in soil when dead plant and animal matter decays .
</item 8>

<item 9>
Humus
The process that converts soil organic matter into humus feeds the population of microorganisms and other creatures in the soil , and thus maintains high and
healthy levels of soil life . The rate at which soil organic matter is converted into humus promotes ( when fast ) or limits ( when slow ) the coexistence of
plants , animals , and microorganisms in the soil . Effective humus and stable humus are additional sources of nutrients for microbes : the former provides a
readily available supply and the latter acts as a longeval storage reservoir . Decomposition of dead plant material causes complex organic compounds to be
slowly oxidized ( lignin - like humus ) or to decompose into simpler forms ( sugars and amino sugars , and aliphatic and phenolic organic acids ) , which are
further transformed into microbial biomass ( microbial humus ) or reorganized , and further oxidized , into humic assemblages ( fulvic and humic acids ) , which
bind to clay minerals and metal hydroxides . The ability of plants to absorb humic substances with their roots and metabolize them has been long debated . There
is now a consensus that humus functions hormonally rather than simply nutritionally in plant physiology .
</item 9>

<item 10>
Humus
This suggests an ambiguous boundary between humus and soil organic matter . While distinct , humus is an integral part of soil organic matter . Humification (
edit ) Microorganisms decompose a large portion of the soil organic matter into inorganic minerals that the roots of plants can absorb as nutrients . This
process is termed `` mineralization '' . In this process , nitrogen ( nitrogen cycle ) and the other nutrients ( nutrient cycle ) in the decomposed organic
matter are recycled . Depending on the conditions in which the decomposition occurs , a fraction of the organic matter does not mineralize , and instead is
transformed by a process called `` humification '' into concatenations of organic polymers . Because these organic polymers are resistant to the action of
microorganisms , they are stable , and constitute humus . This stability implies that humus integrates into the permanent structure of the soil , thereby
improving it . Humification can occur naturally in soil or artificially in the production of compost . Organic matter is humified by a combination of
saprotrophic fungi , bacteria , microbes and animals such as earthworms , nematodes , protozoa , and arthropods .
</item 10>
=== END ITEM id=1370321ec5dc494e ===

=== ITEM id=c88f6df371d4ff33 ===
Gold sentences and facts:
Sentence 0: Dragon Ball Z ( ドラゴンボール ゼット , Doragon Bōru Zetto , commonly abbreviated as DBZ ) is the long - running anime sequel to the Dragon Ball TV series , adapted from the final twenty - six volumes of the Dragon Ball manga written by Akira Toriyama . The manga portion of the series debuted in Weekly Shōnen Jump in October 4 , 1988 and lasted until 1995 ; the anime adaptation premiered in Japan on Fuji Television on April 26 , 1989 , taking over its predecessor 's time slot , and ran until its end on January 31 , 1996 , lasting 291 episodes in Japan , and 276 episodes in the United States originally , although all 291 episodes were later broadcast when content from the first 67 episodes was restored .
  Fact 0.0: Dragon Ball Z (ドラゴンボール ゼット, Doragon Bōru Zetto, commonly abbreviated as DBZ) is the long-running anime sequel to the Dragon Ball TV series.
  Fact 0.1: Dragon Ball Z is adapted from the final twenty-six volumes of the Dragon Ball manga written by Akira Toriyama.
  Fact 0.2: The Dragon Ball Z manga portion debuted in Weekly Shōnen Jump in October 4, 1988.
  Fact 0.3: The Dragon Ball Z manga portion lasted until 1995.
  Fact 0.4: The Dragon Ball Z anime adaptation premiered in Japan on Fuji Television on April 26, 1989.
  Fact 0.5: The Dragon Ball Z anime adaptation took over its predecessor's time slot.
  Fact 0.6: The Dragon Ball Z anime adaptation ended on January 31, 1996.
  Fact 0.7: The Dragon Ball Z anime adaptation lasted 291 episodes in Japan.
  Fact 0.8: The Dragon Ball Z anime adaptation originally lasted 276 episodes in the United States.
  Fact 0.9: All 291 Dragon Ball Z episodes were later broadcast, after content from the first 67 episodes was restored.

Retrieved items:
<item 1>
List of Dragon Ball Z episodes
2001 91 -- 93 ( 76 -- 78 edited ) Super Saiyan Goku 26 000000002001 - 10 - 16 - 0000 October 16 , 2001 94 -- 96 ( 79 -- 81 edited ) Eleventh Hour 27
000000002001 - 11 - 13 - 0000 November 13 , 2001 97 -- 100 ( 82 -- 85 edited ) Fall of a Tyrant 28 000000002001 - 11 - 13 - 0000 November 13 , 2001 101 -- 104 (
86 -- 89 edited ) Namek 's End 29 000000002001 - 12 - 11 - 0000 December 11 , 2001 105 -- 107 ( 90 -- 92 edited ) Black Water Mist 30 000000002002 - 01 - 08 -
0000 January 8 , 2002 108 -- 110 ( 93 -- 95 edited ) Garlic Jr .
</item 1>

<item 2>
List of Dragon Ball Z episodes
2000 71 -- 74 ( 57 -- 60 edited ) The Summoning 20 000000002001 - 05 - 08 - 0000 May 8 , 2001 75 -- 77 ( 61 -- 63 edited ) Frieza Saga Transformation 21
000000002001 - 06 - 26 - 0000 June 26 , 2001 78 -- 81 ( 64 -- 66 edited ) Revealed 22 000000002001 - 07 - 10 - 0000 July 10 , 2001 82 -- 84 ( 67 -- 69 edited )
Death of a Prince 23 000000002001 - 08 - 21 - 0000 August 21 , 2001 85 -- 87 ( 70 -- 72 edited ) Clash 24 000000002001 - 08 - 14 - 0000 August 14 , 2001 88 --
90 ( 73 -- 75 edited ) Desperation 25 000000002001 - 09 - 04 - 0000 September 4 ,
</item 2>

<item 3>
List of Dragon Ball Z episodes
Season Episodes Japanese airdates English airdates Saiyan Saga 39 ( 1 -- 39 ) April 26 , 1989 -- March 7 , 1990 September 13 , 1996 -- September 20 , 1997 (
edited ) June 14 , 2005 -- August 18 , 2005 ( uncut ) Namek and Captain Ginyu Sagas 35 ( 40 -- 74 ) March 14 , 1990 -- January 16 , 1991 September 20 , 1997 --
April 4 , 1998 ( edited ) September 13 , 1999 -- September 20 , 1999 ( uncut ep. 68 -- 74 ) August 22 , 2005 -- October 10 , 2005 ( uncut ep. 40 -- 67 ) Frieza
Saga 33 ( 75 -- 107 ) January 23 , 1991 -- September 11 , 1991 September 21 , 1999 -- November 3 , 1999 Garlic Jr. , Trunks and Androids Sagas 32 ( 108 -- 139 )
September 18 ,
</item 3>

<item 4>
List of Dragon Ball Z episodes
( Saiyan Sized Secret ) `` Ima da Gokū ! Subete o Kaketa Saigo no Ōwaza '' ( いまだ 悟空 ! すべて を 賭け た 最後 の 大 技 ) December 20 , 1989 May 3 , 1997 August 4 , 2005 (
re-dub ) 32 24 `` Battle Power Times Ten ! ! Vegeta 's Great Metamorphosis '' / `` Vegeta ... Saiyan Style ! '' ( Spirit Bomb Away ! ) `` Sentōryoku Jūbai ! !
Bejīta Daihenshin '' ( 戦闘 力 10 倍 ! ! ベジータ 大 変身 ) January 17 , 1990 May 10 , 1997 August 8 , 2005 ( re-dub ) 33 25 `` Do n't Die , Father ! ! This is the Depth
of Gohan 's Power '' / `` Stop Vegeta Now ! ''
</item 4>

<item 5>
List of Dragon Ball Z episodes
July 12 , 1989 October 25 , 1996 June 30 , 2005 ( re-dub ) 12 8 `` Catnap on the Serpentine Road Goku Takes a Tumble '' / `` Home For Infinite Losers '' (
Global Training ) `` Hebi no Michi de Inemuri Gokū ga Okkochiru '' ( 蛇 の 道 で い ねむり 悟空 が 落っこちる ) July 19 , 1989 November 1 , 1996 July 4 , 2005 ( re-dub ) 13 8
`` Hands Off ! Yama - sama 's Secret Fruit '' / `` Home For Infinite Losers '' ( Goz and Mez ) `` Te o Dasu na ! Enmasama no Himitsu no Kudamono '' ( 手 を 出す な !
</item 5>

<item 6>
List of Dragon Ball Z episodes
? '' ( 炎 の 決勝 ! ! 悟空 か パイクーハン か ! ? ) August 25 , 1993 September 13 , 2001 199 184 `` Do n't Let Victory Get Away ! ! Finish With an Ultra-Fast Kamehame - Ha ''
/ `` Goku vs. Pikkon '' `` Nigasu na Shōri ! ! Kimero Chōhaya Kamehameha '' ( 逃す な 勝利 ! ! 決めろ 超速 かめ はめ 波 ) September 1 , 1993 September 14 , 2001 200 185 ``
Seven Years Since Then ! Starting Today , I 'm a High School Student '' / `` Gohan Goes to High School '' `` Are kara Shichinen ! Kyō kara Boku wa Kōkōsei '' (
あれ から 7 年 !
</item 6>

<item 7>
List of Dragon Ball Z episodes
! ブッ 飛べ ヤコン ) April 13 , 1994 October 18 , 2001 224 209 `` A Great Miscalculation ! ! Satan vs. Three Super-Warriors ! ? '' / `` Battle Supreme '' `` Daigosan !
! Satan tai Sannin no Chōsenshi ! ? '' ( 大 誤算 ! ! サタン VS3 人 の 超 戦士 ! ? ) April 20 , 1994 October 22 , 2001 225 210 `` Such Strong Kids ! ! No. 18 's Close Fight
! ? '' / `` Eighteen Unmasks '' `` Tsuyoi ze Chibikko ! ! Jūhachigō Daikusen ! ? '' ( 強い ぜ チビッコ ! ! 18 号 大 苦戦 ! ? )
</item 7>

<item 8>
List of Dragon Ball Z episodes
The series was broadcast in eight separate near - continuous blocks ( 1 - 35 , 36 - 67 , 68 - 107 , 108 - 116 , 117 - 194 , 195 - 237 , 238 - 276 , and 277 -
291 ) , with breaks varying between four months to over a year between each block . In only one instance ( between episodes 194 and 195 ) was there actually
parity between the DVD release and the actual broadcast sequence in terms of the end of one `` season '' and the beginning of the next .
</item 8>

<item 9>
List of Dragon Ball Z episodes
2007 6 75 -- 107 Frieza Saga Season Four February 19 , 2008 6 108 -- 139 Garlic Jr. / Trunks / Androids Saga Season Five May 27 , 2008 6 140 -- 165 Imperfect
Cell / Perfect Cell Saga Season Six September 16 , 2008 6 166 -- 194 Cell Games Saga Season Seven November 11 , 2008 6 195 -- 219 Great Saiyaman / World
Tournament Saga Season Eight February 10 , 2009 6 220 -- 253 Babidi / Majin Buu Saga Season Nine May 19 , 2009 6 254 -- 291 Fusion / Kid Buu Saga Funimation
Dragon Box Z Sets ( edit ) Name Date Discs Episodes Dragon Box Z , Vol. 1 November 17 , 2009 6 1 -- 42 Dragon Box Z , Vol. 2 February 16 , 2010 6 43 -- 84
Dragon Box Z , Vol. 3 May 4 , 2010 6 85 -- 126 Dragon Box Z , Vol.
</item 9>

<item 10>
List of Dragon Ball Z episodes
! ) March 25 , 1992 September 25 , 2000 134 119 `` Too Late to Do Anything ! ? The Ultimate Weapons to Kill Goku '' / `` Goku 's Assassin '' `` Subete ga
Teokure ka ! ? Gokū o Korosu Saishū Heiki '' ( すべて が 手遅れ か ! ? 悟空 を 殺す 最終 兵器 ) April 1 , 1992 September 26 , 2000 135 120 `` Good Looks and Super Power ! ? No
Blind Spot on No. 18 '' / `` Deadly Beauty '' `` Kawaii Kao de Chōpawā ! ? Jūhachigō ni Shikaku Nashi '' ( カワイイ 顔 で 超 パワー ! ? 18 号 に 死角 なし ) April 15 , 1992
September 27 , 2000 136 121 `` Nobody Is Able to Stop Them ... Is This the End of the Z Warriors !
</item 10>
=== END ITEM id=c88f6df371d4ff33 ===
